import numpy as np
import picamera
from picamera.array import PiRGBArray
import cv2
import threading
#import multiprocessing
#from pupil_apriltags import Detector
from dt_apriltags import Detector
import aprilTagLocations as atl
import GPy
import emukit.multi_fidelity 
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
#import dynamicModels
import scipy 
#from scipy.integrate import solve_ivp 
import matplotlib.pyplot as plt
import GraceObservers 
from GraceObservers import kalmanUpdate,kalmanPrediction, eulerToRotm, rot2eul
import GraceRIGV3 as IG
import time
import controllerHelper as ch
import exploreExpSettings as ess

nocontrol=ess.nocontrol
	
def videoRecorder():
	global aprilFrame,recordVidFlag,aprilFrameProcessed,aprilFrameCount,at,frameCount
	VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'H264'),
    #'mp4': cv2.VideoWriter_fourcc(*'XVID'),
	}
	ext='mp4'
	filename=savePath+"vid."+ext
	out = cv2.VideoWriter(filename, VIDEO_TYPE[ext], fps, (resx, resy))
	
	for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame = f.array
		if aprilFrameProcessed:
			at=time.time()
			aprilFrame = frame.copy()
			aprilFrameCount=frameCount
			aprilFrameProcessed=False
			
		if recordVidFlag:
			out.write(frame)
		rawCapture.truncate(0)
		frameCount=frameCount+1
		if not running:
			if recordVidFlag:
				out.release()
			break

def aprilLocator(): #function that maps the apriltags and has some of the setup to estimate the glider position based on previously mapped aprilTags
	global aprilState,running,deltaYaw,aprilFrameProcessed,aprilFrame,aprilFrameCount,at,GPSupdate,alast,aprilVar,useAprilVel
	alast=time.time()
	aprilState=np.zeros((8,1))
	P=atl.Papril
	aprilVar=np.diag(P)
	Q=atl.Qapril
	Rmeas1=atl.Rmeas1april
	Rmeas2=atl.Rmeas2april
	A=atl.Aapril
	rdt=atl.rdtapril
	cdt=atl.cdtapril
	B=atl.Bapril
	imu_in_camera_frame=atl.imu_in_camera_frame
	aprilWorld=np.empty((0,11))
	header1="t,frameCount,id,x,y,z,roll,pitch,yaw,pose_err,mirrored"
	f=open(savePath+"groundTruth.csv",'w')#estimate of robot pose from single tag detections
	f.write(header1+'\n')
	f.close()
	aprilRaw=np.empty((0,9))#world and relative frame poseses
	header1="t,frameCount,id,x,y,z,roll,pitch,yaw"
	f=open(savePath+"aprilRaw.csv",'w')#raw oput of AprilTag detection algorithm
	f.write(header1+'\n')
	f.close()
	aprilFused=np.empty((0,10))
	header1="t,frameCount,x,y,z,yaw,vx,vy,vz,vyaw"
	f=open(savePath+"aprilPressureFusion.csv",'w')#estimate of robot position from Kalma filter using all available data
	f.write(header1+'\n')
	f.close()
	aprilFusedVar=np.empty((0,10))
	header1="t,frameCount,x,y,z,yaw,vx,vy,vz,vyaw"
	f=open(savePath+"aprilPressureFusionVar.csv",'w')#estimate of robot state variance from Kalma filter using all available data
	f.write(header1+'\n')
	f.close()
	#boundaries=[max(atl.tankPoses[:,1]),min(atl.tankPoses[:,1]),max(atl.tankPoses[:,2]),min(atl.tankPoses[:,2])]#xmin,xmax,ymin,ymax
	xmax,xmin,ymax,ymin=atl.boundariesXY#[max(atl.tankPoses[:,1]),min(atl.tankPoses[:,1]),max(atl.tankPoses[:,2]),min(atl.tankPoses[:,2])]#xmin,xmax,ymin,ymax
	windowLen=atl.windowLen
	window=np.ones((3,windowLen))*-1000#time,x,y
	windowTime=atl.windowTime#seconds
	filt_tail=0
	windowFilled=False
	while running:
		if not aprilFrameProcessed:
			deltaYaw=0
			meas=[depth,yaw]
			gray = cv2.cvtColor(aprilFrame, cv2.COLOR_BGR2GRAY)
			#cv2.imwrite("Experiments/"+"{0:.5f}".format(at)+"BS.jpg",aprilFrame)
			mirrored=False
			
			alast=time.time()
			ADD_GPS_MEAS=GPSupdate
			if ADD_GPS_MEAS:
				try:
					GPSx,GPSy,GPSyaw=(float(GPSData[2]),float(GPSData[3]),float(GPSData[4]))
					imc=5#initMeasCount
				except:
					imc=2
					ADD_GPS_MEAS=False
			else:
				imc=2
			tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=t_size)
			if len(tags)==0:
				gray = cv2.cvtColor(cv2.flip(aprilFrame,0), cv2.COLOR_BGR2GRAY)
				tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=t_size)
				mirrored=True
			useAprilVel= len(tags)>0 or ADD_GPS_MEAS
			h=np.zeros((imc+len(tags)*4,8))
			h[0,2]=1
			if len(tags)<1:
				h[1,3]=0#1
			Rmeas=[Rmeas1,Rmeas2[3]]
			if ADD_GPS_MEAS:												
				h[2,0]=1#GPSx
				h[3,1]=1#GPSy
				h[4,3]=1#GPSyaw
				yawGPS=aprilState[3,0]+ch.angleWrap(-aprilState[3,0]+GPSyaw,np.pi) #unwraps angle if near +/-180
				meas.append(GPSx)
				Rmeas.append(atl.GPSxynoise)
				meas.append(GPSy)
				Rmeas.append(atl.GPSxynoise)
				meas.append(yawGPS)
				Rmeas.append(atl.GPSyawnoise)
			
			peScale=1e5
			hn=imc
			
			for tag in tags:
				tid=tag.tag_id
				if not (tid in atl.idMap):
					meas.append(0)
					meas.append(0)
					meas.append(0)
					meas.append(0)
					Rmeas.append(5)
					Rmeas.append(5)
					Rmeas.append(5)
					Rmeas.append(5)
					hn=hn+4
					#continue
				R=tag.pose_R
				p=tag.pose_t
				# get position and orientation measurments
				tag_in_cam_frame=atl.RpToTf(R,p)
				cam_in_tag_frame=np.linalg.inv(tag_in_cam_frame)
				temp=np.insert(atl.tfToVec(tag_in_cam_frame),0,[at,aprilFrameCount,tag.tag_id])
				temp.shape=(1,9)
				aprilRaw=np.append(aprilRaw,temp,axis=0)
				if tid in atl.idMap:# estimate robot pose from known poses
					tag_in_world_frame=atl.idMap[tid]
					cam_in_world_frame=np.matmul(tag_in_world_frame,cam_in_tag_frame)
					imu_in_world_frame=np.matmul(cam_in_world_frame,imu_in_camera_frame)
					
					#filter bad measurements using window average
					x_temp,y_temp=imu_in_world_frame[0:2,3]
					OutOfTank= x_temp>xmax or x_temp<xmin or y_temp>ymax or y_temp<ymin
					#OutOfTank=False
					filt=window[:,np.where(window[0,:]>time.time()-windowTime)[0]]
					reject=False
					if min(filt.shape)!=0:
						if ADD_GPS_MEAS:
							reject=OutOfTank or abs(x_temp-GPSx)>0.35 or abs(y_temp-GPSy)>0.35
						else:#making assumption that mean of the window will be close to true measurement
							reject=abs(filt[1,:].mean()-x_temp)>max(0*filt[1,:].var(),.25) or abs(filt[2,:].mean()-y_temp)>max(0*filt[2,:].var(),.25) or OutOfTank
					if 1:# not reject or not windowFilled:
						window[:,filt_tail]=[at,x_temp,y_temp]
						filt_tail=(filt_tail+1)%windowLen
					if filt_tail==windowLen-1:
						windowFilled=True
					elif len(filt[0,:])<5:
						windowFilled=False
					if reject and windowFilled:
						mirrored=-100
						
					aprilpose=np.insert(atl.tfToVec(imu_in_world_frame),0,[at,aprilFrameCount,tid])  
					aprilpose=np.append(aprilpose,[tag.pose_err,mirrored],axis=0)
					meas.append(aprilpose[3])
					meas.append(aprilpose[4])
					if mirrored:
						meas.append(depth)
					else:
						meas.append(aprilpose[5])
					#meas.append(aprilpose[7])
					meas.append(aprilState[3,0]+ch.angleWrap(-aprilState[3,0]+aprilpose[8],np.pi))
					
					if not reject or not windowFilled:
						h[hn:hn+4,:4]=np.eye(4)
					hn=hn+4
					#scale=np.sqrt((cx-2*tag.center[0])**2+(cy-2*tag.center[1])**2)/np.sqrt(resx**2+resy**2)
					scale=np.linalg.norm(p)
					skewness=atl.tfToVec(imu_in_world_frame)
					scale2=3*np.sqrt((skewness[3])**2+(skewness[4])**2)/2.22144
					highNoiseTag=1*(tid in [])
					Rmeas.append(Rmeas2[0]*(1+scale+scale2+tag.pose_err*peScale)+highNoiseTag)
					Rmeas.append(Rmeas2[1]*(1+scale+scale2+tag.pose_err*peScale)+highNoiseTag)
					Rmeas.append(Rmeas2[2]*(1+scale+scale2+tag.pose_err*peScale))
					Rmeas.append(Rmeas2[3]*(1+scale+scale2+tag.pose_err*peScale/100))
					aprilWorld=np.append(aprilWorld,[aprilpose],axis=0)
					
			
					
				
			
			dt=time.time()-alast		
				
					
			A[rdt,cdt]=dt 
			aprilState,P=kalmanUpdate(aprilState,P,np.array([meas]).transpose(),h,np.diag(Rmeas))
			aprilState[3,0]=np.mod(aprilState[3,0]+np.pi,2*np.pi)-np.pi
			   
			#aprilState,P=kalmanPrediction(aprilState,np.array([deltaYaw]),A,B,P,Q)
			aprilState,P=kalmanPrediction(aprilState,0,A,0,P,Q)
			aprilState[3,0]=np.mod(deltaYaw+aprilState[3,0]+np.pi,2*np.pi)-np.pi
			aprilVar=np.diag(P)
			aprilFrameProcessed=True
			GPSupdate=False		
			
			temp=aprilState[:,0].tolist()
			temp=np.insert(temp,0,[time.time(),aprilFrameCount])
			aprilFused=np.append(aprilFused,[temp],axis=0)	
			temp=np.diag(P).tolist()
			temp=np.insert(temp,0,[time.time(),aprilFrameCount])
			aprilFusedVar=np.append(aprilFusedVar,[temp],axis=0)	
			
				
			if max(aprilFused.shape)>500:
				f=open(savePath+"aprilPressureFusion.csv",'a')
				np.savetxt(f,aprilFused,delimiter=',')
				f.close()
				aprilFused=np.empty((0,aprilFused.shape[1]))	
			if max(aprilFusedVar.shape)>500:
				f=open(savePath+"aprilPressureFusionVar.csv",'a')
				np.savetxt(f,aprilFusedVar,delimiter=',')
				f.close()
				aprilFusedVar=np.empty((0,aprilFusedVar.shape[1]))
			if max(aprilRaw.shape)>500:
				f=open(savePath+"aprilRaw.csv",'a')
				np.savetxt(f,aprilRaw,delimiter=',')
				f.close()
				aprilRaw=np.empty((0,aprilRaw.shape[1]))
			if max(aprilWorld.shape)>500:
				f=open(savePath+"groundTruth.csv",'a')
				np.savetxt(f,aprilWorld,delimiter=',')
				f.close()
				aprilWorld=np.empty((0,aprilWorld.shape[1]))
			if not running:
				f=open(savePath+"groundTruth.csv",'a')
				np.savetxt(f,aprilWorld,delimiter=',')
				f.close()
				f=open(savePath+"aprilPressureFusion.csv",'a')
				np.savetxt(f,aprilFused,delimiter=',')
				f.close()
				f=open(savePath+"aprilRaw.csv",'a')
				np.savetxt(f,aprilRaw,delimiter=',')
				f.close()
				f=open(savePath+"aprilPressureFusionVar.csv",'a')
				np.savetxt(f,aprilFusedVar,delimiter=',')
				f.close()
				break
				
	
	
def xbeeListener():
	global running,GPSupdate,GPSData
	header="t,frame,mlabel,tGPS,available,x,y,yaw"
	fGPS=open(savePath+"GPS.csv",'w')
	fGPS.write(header+'\n')
	fGPS.close()
	fGPS=open(savePath+"GPS.csv",'a')
	while running:
		data=xbee.recv(1024)
		msg=data.decode("utf-8")
		#print(msg)
		info=msg.split(',')
		#print(info)
		if "BEGIN" in info[1]:
			print("start")
		elif "STOP" in info[1]:
			running=False
			print("stopped")
		elif "SNAP" in info[1]:
			print("took snapshot")
		elif "CAMWPT" in info[1]:#OBTTC,CAMWPT,
			pass#print(info)
		elif "CameraGPS" in info[1]:#OBTTC,CameraGPS,time,reliable,x,y,yaw
			try:
				GPSData=info[2:]#time,reliable,x,y,yaw 
				(float(GPSData[2]),float(GPSData[3]),float(GPSData[4]))
				GPSupdate=info[3]=="True"
				fGPS.write("{},{},{}".format(time.time(),frameCount,msg.replace(info[0]+",","")))
			except:
				pass
	fGPS.close()
		
	
	
	
running=True
GPSupdate=False
xbee=ch.connectToServer('./XBEE_NODE')
xbee.send(bytes("updateMe",'utf-8')) #msg to send from GPS computer: OBTTC,MsgHeadr,data

aprilFrameProcessed=True	
recordVidFlag=True
aprilFrame=None

GPSData=[]
savePath="Data/"
params=np.loadtxt(cfg.paramPath,delimiter=',')
paramsFlat=np.loadtxt(cfg.paramPath,delimiter=',')
paramsSwim=np.loadtxt("calibrationData/params_swim.model",delimiter=',')
#print(params)


np.savetxt(savePath+"model_params.csv",params,delimiter=',')


#resx,resy=(320,240)
resx,resy=(640,480)
#resx,resy=(720,480)
#resx,resy=(800,608)
#resx,resy=(1088,720)

fps=10
camera = picamera.PiCamera()              #Camera initialization
camera.resolution = (resx, resy)
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=(resx, resy))
frameCount=0

t_size=13.6/100 # m
fx,fy,cx,cy = (608.14,609.30,322.16,234.34)
f=open(savePath+"aprilParams.csv",'w')
f.write("tagSize={0} meters\ncamres=({1},{2}),(fx,fy,cx,cy)=({3},{4},{5},{6}))".format(t_size,resx,resy,fx,fy,cx,cy))
f.close()


	

at_detector = Detector(families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

IMUSock = ch.connectToServer('./IMU')
I2CSock = ch.connectToServer('./I2C_NODE')
RGBSock = ch.connectToServer('./ARDU_NODE')
LEDSock = ch.connectToServer('./LED_NODE')
time.sleep(5)
########################## controller setup ###############
tail=ch.Swimming(0,0,.75)
tail.wave="sin"
#################################### initial setup ########
aprilState=np.zeros((8,1))

depthOffset=0
tstart=0
tlast_ctrl=tstart
tlast=tstart

GPSThread=threading.Thread(target=xbeeListener)
GPSThread.start()


tstart=time.time()


roll,pitch,yaw2 = ch.readEuler(IMUSock)
yaw=float(GPSData[4]) if GPSupdate else 0
deltaYaw=0
RotmApril = atl.eul2rotm([roll,pitch,yaw]) 
gx,gy,gz=ch.readGyro(IMUSock)
ax,ay,az=ch.readAccel(IMUSock)
trgb,red,green,blue=ch.readRGB(RGBSock)
u=ch.readInputs(I2CSock)
battV=ch.readBattVolt(I2CSock)
depthOffset=-ch.readDepth(IMUSock)
depth=0
at=time.time()
camThread=threading.Thread(target=videoRecorder)
camThread.start()
useAprilVel=False
aprilThread=threading.Thread(target=aprilLocator)
aprilThread.start()

u=[u[0]/100,u[1]/100,np.deg2rad(u[2])]#mass,pump,tail



time.sleep(2)



#setup kalman filter for position. state: x,y,z,z_dot
Pxhat=ess.Pxhat2
Qxhat=ess.Qxhat2
Axhat=ess.Axhat2
 	
Rxhat=ess.Rxhat2
xhat=np.array([[aprilState[0,0]],[aprilState[1,0]],[depth],[0],[0],[0]]) #kf position and velocity estimate
Phat=np.array([[aprilState[0,0]],[aprilState[1,0]],[depth]])  #nonlinear observer position estimate
#input speed kf. state: mass position,pump position, mass speed, pump speed
Pinp=ess.Pinp
Qinp=ess.Qinp
Ainp=ess.Ainp
Rinp=ess.Rinp
inpHat=np.array([[u[0]],[u[1]],[0],[0]]) #kf actuator state and rate estimate


vb_est=np.array([[1e-4],[0],[1e-4]]) 
estimationVars=np.array([[tstart,frameCount,Phat[0,0],Phat[1,0],Phat[2,0],xhat[0,0],xhat[1,0],xhat[2,0],xhat[3,0],xhat[4,0],xhat[5,0],Pxhat[0,0],Pxhat[1,1],Pxhat[2,2],Pxhat[3,3],Pxhat[4,4],Pxhat[5,5],vb_est[0,0],vb_est[1,0],vb_est[2,0],0,0,0,0,0,0]])
measVars=np.array([[tstart,frameCount,u[0], u[1], u[2], depth, roll, pitch, yaw,yaw2,0,0, gx, gy, gz,ax ,ay, az,battV,trgb,red,green,blue]])
cntrlVars=np.array([[tstart,frameCount,0,0,0,0,0,0,0,0,0,0,0,0]])

#function to interpolate (linear) 3D path
trajPnt= ess.trajPnt

WS=ess.WS		   
maxDepth=ess.maxDepth
FS=WS

fidlevels=ess.fidlevels
df3=[tstart]
df2=[tstart]
df1=[tstart]
lastSampleTime=tstart
Xhf3=estimationVars[np.isin(estimationVars[:,0],df3),2:5]
Xhf2=estimationVars[np.isin(estimationVars[:,0],df2),2:5]
Xhf1=estimationVars[np.isin(estimationVars[:,0],df1),2:5]
y3=measVars[np.isin(measVars[:,0],df3),-1]
y3.shape=(y3.shape[0],1)
y2=measVars[np.isin(measVars[:,0],df2),-1]
y2.shape=(y2.shape[0],1)
y1=measVars[np.isin(measVars[:,0],df1),-1]
y1.shape=(y1.shape[0],1)


Xs=estimationVars[np.isin(estimationVars[:,0],df1+df2+df3),2:5]
Xhs=estimationVars[np.isin(estimationVars[:,0],df1+df2+df3),2:5]
ys=measVars[np.isin(measVars[:,0],df1+df2+df3),-1]
ys.shape=(ys.shape[0],1)

#mean=GPy.mappings.Constant(3,1)
kernel=GPy.kern.RBF(input_dim=3,variance=1,lengthscale=1,ARD=True)
#gp=GPy.models.GPRegression(Xhs,ys,kernel,mean_function=mean)
gp=GPy.models.GPRegression(Xhs,ys,kernel)
HypNames=gp.parameter_names()
f=open("Data/GPySFGP.txt","w")
f.write("rbf_var,rbf_lx,rbf_ly,rbf_lz,noise\n")
f.write(str(HypNames)+"\n")
f.close()
HypVec=gp.param_array.copy()
f=open("Data/GPySFGP.txt","a")
HypVec.shape=(1,HypVec.shape[0])
np.savetxt(f,HypVec,delimiter=",")
f.close()

#setup RIG planner
agent=ess.agent
agent.CalcCost=agent.calcPathInfoSF 											  
agent.sfgp=gp
print("Max time underwater: ",agent.underWaterTimeLimit)

BudgetUsed=0
PlannedBudget=0
udot_weights=ess.udot_weights
delta_hat=u[2]
k_delta=paramsSwim[31]#max vel=paramsSwim[32]



useGPS=False

tstart=time.time()
tlast_ctrl=tstart
tlast=tstart
u1=u2=0
uc=[0,0,0]
toff=0
maxBlue=blue
while running:#time.time()-tstart<60*8:# main loop to collect sensor data and estimate state of miniglider
	
	dt=time.time()-tlast
	tlast=time.time() 
	roll,pitch,yaw2 = ch.readEuler(IMUSock)
	ax,ay,az=ch.readAccel(IMUSock)
	gx,gy,gz=ch.readGyro(IMUSock)
	tempDepth=ch.readDepth(IMUSock)
	depth=tempDepth+depthOffset if abs(tempDepth)<10 else depth
	lastmass=u[0]
	u=ch.readInputs(I2CSock)
	battV=ch.readBattVolt(I2CSock)
	trgb,red,green,blue=ch.readRGB(RGBSock)
	if blue<0:
		print("rgb sensor malfunction: ",blue)								
		running=False
	u=[u[0]/100,u[1]/100,np.deg2rad(u[2])]#mass,pump,tail
	if u[0]>1 or u[0]<0:
		u[0]=lastmass
	#input estimation stuff
	#ddelta=ch.saturate(k_delta*(u[2]-delta_hat),-paramsSwim[32],paramsSwim[32])
	ddelta=k_delta*ch.saturate((u[2]-delta_hat),-np.pi,np.pi)
	delta_hat=ch.saturate(delta_hat+(ddelta*dt-.5*k_delta*ddelta*dt**2)*(dt<1),-np.deg2rad(110),np.deg2rad(110))
	#delta_hat=delta_hat+ddelta*dt-.5*k_delta*ddelta*dt**2
	inpHat,Pinp=kalmanPrediction(inpHat,0,Ainp(dt),0,Pinp,Qinp*dt)
	inpHat,Pinp=kalmanUpdate(inpHat,Pinp,np.array([[u[0]],[u[1]]]),ess.Hinp,Rinp)
	_,_,dmass,dpump=inpHat.flatten().tolist()
	udot=np.array([dmass**2,dpump**2,ddelta**2,agent.timeEnergy])
	BudgetUsed=BudgetUsed+np.sum(udot*udot_weights)*dt
	
	#assign time stamps to pull data for multifidelity model
	
	if time.time()-lastSampleTime>1/agent.measRate or (blue>ess.blueThresh*maxBlue and time.time()-lastSampleTime>.25/agent.measRate):
		if blue>maxBlue:
			maxBlue=blue
		lastSampleTime=time.time()
		covComp=np.trace(Pxhat[0:2,0:2])
		if covComp <fidlevels[0]:
			df1.append(tlast)
		elif covComp<fidlevels[1]:
			df2.append(tlast)
		else:#elif covComp<fidlevels[2]:
			df3.append(tlast)
		
	useGPS=depth<ess.atSurface			
	tuav=useAprilVel and (time.time()-alast)<1					  
	meas_vec=[0,  0, depth,  0,  0,  0,  gx,gy,gz] 		
	if useAprilVel:
		yaw=aprilState[3,0]
	if useGPS and useAprilVel:
		meas_vec[0]=aprilState[0,0]
		meas_vec[1]=aprilState[1,0]
		Phat[0,0]=aprilState[0,0]
		Phat[1,0]=aprilState[1,0]
	elif useGPS:
		try:
			GPSx,GPSy=(float(GPSData[2]),float(GPSData[3]))
			meas_vec[0]=GPSx
			meas_vec[1]=GPSy
			Phat[0,0]=GPSx
			Phat[1,0]=GPSy
		except:
			pass
	#print(roll,pitch,yaw)
	Rotm = atl.eul2rotm([roll,pitch,yaw])
	r=(gy*np.sin(roll)+gz*np.cos(roll))/np.cos(pitch)*dt

	yaw=yaw+r
	deltaYaw=deltaYaw+r
	dP,dvb=GraceObservers.velEstimator(meas_vec,Rotm,u,vb_est,Phat[2,0],params) #dp is vector velocity in xyz. 
	#dvb derivative of body fixed velocity. velocity with respect to the center of the glider.
	Phat=Phat+dP*dt
	vb_est=vb_est+dvb*dt*(dt<.5) 
	if np.isnan(vb_est).any():
		Phat[0:3,0]=np.array([aprilState[0,0],aprilState[1,0],depth]) 
		vb_est=np.array([[1e-4],[0],[1e-4]]) 
		print("velocity observer singularity")
	useVel_obs=not (dt>.5)*np.isnan(vb_est).any() and ddelta<np.rad2deg(10)
	vel_obs=np.matmul(Rotm,vb_est)
	Hxhat=np.diag([useGPS*tuav,useGPS*tuav,1,tuav,tuav,tuav])
	Hxhat=np.vstack((Hxhat,np.diag([useVel_obs,useVel_obs,useVel_obs],k=3)[:3,:]))
	kfmeas=np.array([[aprilState[0,0]],[aprilState[1,0]],[depth],[aprilState[4,0]],[aprilState[5,0]],[aprilState[6,0]],[vel_obs[0,0]],[vel_obs[1,0]],[vel_obs[2,0]]])																			   
	tx,ty,tz=aprilVar[4:7]
	Rxhat[5,5]=ess.velVarMult*tz	#vz															  
	Rxhat[4,4]=ess.velVarMult*ty	#vy															  
	Rxhat[3,3]=ess.velVarMult*tx	#vx																	  
	xhat,Pxhat=kalmanPrediction(xhat,0,Axhat(dt),0,Pxhat,Qxhat*dt)
	xhat,Pxhat=kalmanUpdate(xhat,Pxhat,kfmeas,Hxhat,Rxhat)
	xhat[0,0]=ch.saturate(xhat[0,0],ess.xmin,ess.xmax)
	xhat[1,0]=ch.saturate(xhat[1,0],ess.ymin,ess.ymax)
	
	
	
	
				
	estimationVars=np.append(estimationVars,[[tlast,frameCount,Phat[0,0],Phat[1,0],Phat[2,0],xhat[0,0],xhat[1,0],xhat[2,0],xhat[3,0],xhat[4,0],xhat[5,0],Pxhat[0,0],Pxhat[1,1],Pxhat[2,2],Pxhat[3,3],Pxhat[4,4],Pxhat[5,5],vb_est[0,0],vb_est[1,0],vb_est[2,0],dvb[0,0],dvb[1,0],dvb[2,0],BudgetUsed,PlannedBudget,0]],axis=0) 
	measVars=np.append(measVars,[[tlast,frameCount,u[0], u[1], u[2], depth, roll, pitch, yaw,yaw2,r,0, gx, gy, gz,ax ,ay, az,battV, trgb, red,green,blue]],axis=0) 
	cntrlVars=np.append(cntrlVars,[[tlast,frameCount,u2,u1,tail.bias,tail.amp,tail.freq,uc[0],uc[1],uc[2],dmass,dpump,delta_hat,ddelta]],axis=0) 
	
	
		
		#save data	
	if max(estimationVars.shape)>1000:
		f=open(savePath+"estimates.csv",'a')
		np.savetxt(f,estimationVars,delimiter=',')
		f.close()
		estimationVars=np.empty((0,estimationVars.shape[1]))
	if max(cntrlVars.shape)>1000:
		f=open(savePath+"control.csv",'a')
		np.savetxt(f,cntrlVars,delimiter=',')
		f.close()
		cntrlVars=np.empty((0,cntrlVars.shape[1]))
	if max(measVars.shape)>1000:
		f=open(savePath+"measurements.csv",'a')
		np.savetxt(f,measVars,delimiter=',')
		f.close()
		measVars=np.empty((0,measVars.shape[1]))
	time.sleep(1/1000)
	#if time.time()-trajTimeStart>totalTime:
	#	break
		
running=False
tail.stop()

f=open(savePath+"measurements.csv",'a')
np.savetxt(f,measVars,delimiter=',')
f.close()
f=open(savePath+"estimates.csv",'a')
np.savetxt(f,estimationVars,delimiter=',')
f.close()  
f=open(savePath+"control.csv",'a')
np.savetxt(f,cntrlVars,delimiter=',')
f.close()  


LEDSock.send("random".encode('utf-8'))	
time.sleep(1)
LEDSock.send("off".encode('utf-8'))
xbee.send(bytes("done",'utf-8'))
xbee.send(bytes("stopUpdates",'utf-8'))
print("Done...")
#msg to send from control computer to stop video: OBTTC,STOP

#load data
estData=np.loadtxt("Data/estimates.csv",skiprows=1,delimiter=",")
measData=np.loadtxt("Data/measurements.csv",skiprows=1,delimiter=",")
print("Blue stats: mean=",np.mean(measData[:,-1]),", max=",np.max(measData[:,-1]),", min=",np.min(measData[:,-1]))
GPDataFile="Data/GPData{0}.csv".format(0)
GPDataPointers=np.array([df1+df2+df3]).T
fidLevs=np.concatenate((0*np.ones((len(df1),1)),1*np.ones((len(df2),1)),2*np.ones((len(df3),1))),axis=0)
np.savetxt(GPDataFile, np.concatenate((GPDataPointers,fidLevs),axis=1),delimiter=",",header="t,fid",comments="")


#extract relevant stuff for GP and train
# train a dingle fidelity model
Xhs=estData[np.isin(estData[:,0],df1+df2+df3),5:8]
ys=ess.ftf(measData[np.isin(measData[:,0],df1+df2+df3),-1])
ys.shape=(ys.shape[0],1)
gp.set_XY(Xhs,ys)
#gp.optimize()
gp._save_model("Data/GPmodel",compress=False)
'''
HypVec=gp.param_array.copy()
f=open("Data/GPySFGP.txt","a")
HypVec.shape=(1,HypVec.shape[0])
np.savetxt(f,HypVec,delimiter=",")
f.close()	
# train a multi fidelity model
Xhf3=estData[np.isin(estData[:,0],df3),5:8]
Xhf2=estData[np.isin(estData[:,0],df2),5:8]
Xhf1=estData[np.isin(estData[:,0],df1),5:8]
y1=measData[np.isin(measData[:,0],df1),-1]
y2=measData[np.isin(measData[:,0],df2),-1]
y3=measData[np.isin(measData[:,0],df3),-1]
y3.shape=(y3.shape[0],1)
y2.shape=(y2.shape[0],1)
y1.shape=(y1.shape[0],1)
Xhs=[Xhf3,Xhf2,Xhf1]
ys=[y3,y2,y1]

#X_train, Y_train = convert_xy_lists_to_arrays([Xf3, Xf2, Xf1], [y3, y2, y1])
Xh_train, Y_train = convert_xy_lists_to_arrays(Xhs,ys)

n_fids=3
kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel, n_fidelities=n_fids)
lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)

lin_mf_model.set_data(Xh_train,Y_train)
lin_mf_model.optimize()
emuHypVec=lin_mf_model.gpy_model.param_array.copy()
print(emuHypVec,emuHypVec.shape)
f=open("Data/emuGP.txt","a")
f.write("rbf_var,rbf_lx,rbf_ly,rbf_lz,rbf1_var,rbf1_lx,rbf1_ly,rbf1_lz,rbf2_var,rbf2_lx,rbf2_ly,rbf2_lz,rho1,rho2,noise,noise1,noise2\n")
f.write(str(in_mf_model.gpy_model.parameter_names())+"\n")
emuHypVec.shape=(1,emuHypVec.shape[0])
np.savetxt(f,emuHypVec,delimiter=",")
f.close()

#predict field
testPoints = ess.testPoints
mu,sig=gp.predict(testPoints)
mumf,sigmf=lin_mf_model.predict(np.hstack((testPoints,2*np.ones((testPoints.shape[0],1)))))

np.savetxt("Data/results.csv",np.concatenate((testPoints,mu,sig,mumf,sigmf),axis=1),delimiter=",",header=" x,y,z,gpMean,gpVar,mfgpMean,mfgpVar",comments="")
'''
