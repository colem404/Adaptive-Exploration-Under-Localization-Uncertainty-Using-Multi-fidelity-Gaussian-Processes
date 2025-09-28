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
import backsteppingConfig as cfg
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

def planWatcher():
	global PlannedBudget,toff,running,planNum,estimationVars,measVars,savingData
	while running:
		if t>path[-1][3]+toff:
			tail.amp=0
			if not useGPS:
				xbee.send(bytes("waiting to plan. depth: "+str(depth)+"\n",'utf-8'))
				if not nocontrol:
					ch.setPumpPos(I2CSock,75)
					time.sleep(0.05)
					ch.setMassPos(I2CSock,45.5)
			else:
				savingData.acquire()
				f=open(savePath+"estimates.csv",'a')
				np.savetxt(f,estimationVars,delimiter=',')
				f.close()
				estimationVars=np.empty((0,estimationVars.shape[1]))
				f=open(savePath+"measurements.csv",'a')
				np.savetxt(f,measVars,delimiter=',')
				f.close()
				savingData.release()
				measVars=np.empty((0,measVars.shape[1]))
				print("Budget used :",BudgetUsed,PlannedBudget)
				if not nocontrol:
					ch.setPumpPos(I2CSock,ess.pumpStart)
					time.sleep(0.05)
					ch.setMassPos(I2CSock,ess.massStart)
				planNum+=1
				 
				#if B-max(BudgetUsed,PlannedBudget-ess.pathEpsilon)<0:
				#if ((B-budgetUsed)<(0.5*B/BD) and planNum>BD):
				if (B-PlannedBudget)<(0.5*B/BD) and planNum>BD:
				#if B-(PlannedBudget+ess.pathEpsilon)<0:
					running=False
				else:
					agent.stopWatchTime=None
					xbee.send(bytes("planning"+str(planNum)+"\n",'utf-8'))
					replan()
					xbee.send(bytes("done planning\n",'utf-8'))
					agent.stopWatchTime=None				  
					PlannedBudget=PlannedBudget+planner.bestPath[0]
					#print(len(planner.V))
					if len(planner.V)>1:
						toff=time.time()-trajTimeStart
					else:
						PlannedBudget=PlannedBudget+ess.planningCost
		time.sleep(1)
	
def replan():#(agent,df1,df2,df3,planNum,GPpreTrained=None):		
	global edgeChain,pathPoints,planner,totalTime,allPathPoints,lin_mf_model
	#load saved data
	print("planning")
	
	estData=np.loadtxt(savePath+"estimates.csv",skiprows=1,delimiter=",")
	measData=np.loadtxt(savePath+"measurements.csv",skiprows=1,delimiter=",")
	print("Blue stats: mean=",np.mean(measData[:,-1]),", max=",np.max(measData[:,-1]),", min=",np.min(measData[:,-1]))
	GPDataFile=savePath+"GPData{0}.csv".format(planNum)
	GPDataPointers=np.array([df1+df2+df3]).T
	fidLevs=np.concatenate((0*np.ones((len(df1),1)),1*np.ones((len(df2),1)),2*np.ones((len(df3),1))),axis=0)
	np.savetxt(GPDataFile, np.concatenate((GPDataPointers,fidLevs),axis=1),delimiter=",",header="t,fid",comments="")
	#extract relevant stuff for GP and train
	Xhf3=estData[np.isin(estData[:,0],df3),5:8]
	Xhf2=estData[np.isin(estData[:,0],df2),5:8]
	Xhf1=estData[np.isin(estData[:,0],df1),5:8]
	y3=ess.ftf(measData[np.isin(measData[:,0],df3),-1])
	y3.shape=(y3.shape[0],1)
	y2=ess.ftf(measData[np.isin(measData[:,0],df2),-1])
	y2.shape=(y2.shape[0],1)
	y1=ess.ftf(measData[np.isin(measData[:,0],df1),-1])
	y1.shape=(y1.shape[0],1)
	
	Xhs=[Xhf3,Xhf2,Xhf1]
	ys=[y3,y2,y1]
	print(Xhf3.shape,Xhf2.shape,Xhf1.shape)
	print(y3.shape,y2.shape,y1.shape)
	#update gp model and save hyperparams
	Xh_train, Y_train = convert_xy_lists_to_arrays(Xhs,ys)	
	lin_mf_model.set_data(Xh_train,Y_train)
	if ess.updateGPHyps:
		print("training GP")
		try:
			last_params=lin_mf_model.gpy_model.param_array
			lin_mf_model.optimize()
		except Exception as e:
			print(e)
			lin_mf_model.gpy_model.param_array[:]=last_params[:]
			try:
				lin_mf_model.optimize()
			except:
				pass
		if np.any(lin_mf_model.gpy_model.param_array>90):
			tempParams=lin_mf_model.gpy_model.param_array.copy()
			tempParams[tempParams>90]=1	
			kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
			lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
			lik=GPy.likelihoods.Gaussian()
			gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel,likelihood=lik, n_fidelities=n_fids)
			lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)
			lin_mf_model.gpy_model.kern.scale.fix([1,1])
			lin_mf_model.gpy_model.kern.rbf.lengthscale.constrain_bounded(0.0001,100)
			lin_mf_model.gpy_model.kern.rbf_1.lengthscale.constrain_bounded(0.0001,100)
			lin_mf_model.gpy_model.kern.rbf_2.lengthscale.constrain_bounded(0.0001,100)
			lin_mf_model.gpy_model.param_array[:]=tempParams[:]
	if USE_SF_IG:
		tempXhs=estData[np.isin(estData[:,0],df1+df2+df3),5:8]
		tempys=measData[np.isin(measData[:,0],df1+df2+df3),-1]
		tempys.shape=(tempys.shape[0],1)
		agent.sfgp.param_array[:-1]=lin_mf_model.gpy_model.param_array.copy()[:4]
		agent.sfgp.param_array[-1]=lin_mf_model.gpy_model.param_array.copy()[-3]
		agent.sfgp.set_XY(tempXhs,tempys)
		print("sf params:",agent.sfgp.param_array)	
	emuHypVec=lin_mf_model.gpy_model.param_array.copy()
	print(emuHypVec,emuHypVec.shape)
	f=open(savePath+"emuGP.csv","a")
	emuHypVec.shape=(1,emuHypVec.shape[0])
	np.savetxt(f,emuHypVec,delimiter=",")
	f.close()
	
	agent.mfgp=lin_mf_model
	
	#generate new plan
	xstart=np.array([[aprilState[0,0]],[aprilState[1,0]]])
	#Btemp=min(B/BD,B-BudgetUsed)
	Btemp=min(B/BD,B-PlannedBudget)
	planner=IG.Graph(stepSize,Btemp,WS,FS,measFunc,nearRad,agent=agent)
	planner.AllowSelfLoops=False
	planner.animate = False
	planner.ModularCost=False
	planner.animateNewEdge = False
	planner.debugMode = False	
	planner.terminalCond=agent.stopWatch
	#planner.maxIter=100
	planner.SameNodeDistance=ess.SameNodeDistance
	planner.EvaluateCost=agent.getCost
	startPlanning=time.time()
	planner.plan(xstart,Rd=ess.Rd)
	planningTime=time.time()-startPlanning
	print("planning time:",planningTime)
	pathBudget,pathInfo,bnode_idx,bpath_idx=planner.bestPath
											 
	if  bpath_idx==None:
		return
	path=planner.V[bnode_idx].pathList[bpath_idx]
	print("Best Path:",planner.bestPath,"\n\ttime: {3}\n\tBudget: {4}\n\toptimization objective: {5}\n".format(*path[-1]))
	#newPathPoints=np.concatenate((pathPoints,agent.pathToTrajPoints(planner.V,planner.E,path,t_off=time.time()-trajTimeStart)))
	newPathPoints=agent.pathToTrajPoints(planner.V,planner.E,path,t_off=time.time()-trajTimeStart)
	totalTime=path[-1][3]+time.time()-trajTimeStart
	edgeChain2=[planner.E[data[0:2]][data[2]] for data in path]
	newEdgeChain=[]
						 
	for data in path:
		for prim in planner.E[data[0:2]][data[2]][-1]:
			newEdgeChain.append(prim)
	print("new edge chain:",newEdgeChain)	

	
	if len(planner.V)>1:
		pathPoints=newPathPoints
		edgeChain=newEdgeChain
	
		allPathPoints=np.vstack((allPathPoints,np.hstack((pathPoints,np.ones((pathPoints.shape[0],1))*planNum))))
	try:
		print("first prim",edgeChain[min(len(edgeChain)-1,max(0,np.sum(time.time()-trajTimeStart>pathPoints[:,3])-1))])
	except:
		print(min(len(edgeChain)-1,max(0,np.sum(time.time()-trajTimeStart>pathPoints[:,3])-1))+1,len(edgeChain))																									  
								  		  
	f=open(savePath+"plannedTraj{0}.csv".format(planNum),'w')
	f.write("x,y,z,t\n")
	np.savetxt(f,newPathPoints,delimiter=',')
	f.close()
	f=open(savePath+"bestPath{0}.txt".format(planNum),'w')
	f.write(str(planner.bestPath)+"\n")
	f.write(str(path)+"\n")
	f.write(str(edgeChain)+"\n")
	f.write(str(agent.CalcCost)+"\n")
	f.write("planning time: "+str(planningTime)+"\n")
	f.close()

	#print(edgeChain)
	planner.node_loc_dict(save=True,fname=savePath+"graphNodes{0}.txt".format(planNum))
	planner.edge_dict(save=True,fname=savePath+"graphEdges{0}.txt".format(planNum))

		
	
	
running=True
GPSupdate=False
xbee=ch.connectToServer('./XBEE_NODE')
xbee.send(bytes("updateMe",'utf-8')) #msg to send from GPS computer: OBTTC,MsgHeadr,data

aprilFrameProcessed=True	
recordVidFlag=True
aprilFrame=None

GPSData=[]
savePath="Data/"
params=ess.params
paramsFlat=ess.paramsFlat
paramsSwim=ess.paramsSwim


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
if not nocontrol:
	ch.setPumpPos(I2CSock,ess.pumpStart)#100*params[2])
time.sleep(0.05)
if not nocontrol:
	ch.setMassPos(I2CSock,ess.massStart)#100*params[0])
controlRate=ess.controlRate#hz
pitchControlRate=ess.pitchControlRate										   
dkp,dkd=ess.linearDepthGains
pkp,pkd=ess.linearPitchGainsp
																										 
																											
tail=ch.Swimming(0,0,.75)
tail.wave="sin"
if not nocontrol:
	tail.run(I2CSock)
################### initial setup ########
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

pitchKF=np.array([[pitch],[0]])
PpitchKF=0.0175*np.diag([3,15])

vb_est=np.array([[1e-4],[0],[1e-4]]) 
estimationVars=np.array([[tstart,frameCount,Phat[0,0],Phat[1,0],Phat[2,0],xhat[0,0],xhat[1,0],xhat[2,0],xhat[3,0],xhat[4,0],xhat[5,0],Pxhat[0,0],Pxhat[1,1],Pxhat[2,2],Pxhat[3,3],Pxhat[4,4],Pxhat[5,5],vb_est[0,0],vb_est[1,0],vb_est[2,0],0,0,0,0,0,1]])
measVars=np.array([[tstart,frameCount,u[0], u[1], u[2], depth, roll, pitch, yaw,yaw2,0,0, gx, gy, gz,ax ,ay, az,battV,trgb,red,green,blue]])
cntrlVars=np.array([[tstart,frameCount,0,0,0,0,0,0,0,0,0,0,0,0]])
#function to interpolate (linear) 3D path
trajPnt= ess.trajPnt

WS=ess.WS							 
										 
maxDepth=ess.maxDepth
FS=WS

fidlevels=ess.fidlevels
df1=[tstart]
df2=[tstart]
df3=[tstart]
lastSampleTime=tstart
Xhf3=estimationVars[np.isin(estimationVars[:,0],df3),2:5]
Xhf2=estimationVars[np.isin(estimationVars[:,0],df2),2:5]
Xhf1=estimationVars[np.isin(estimationVars[:,0],df1),2:5]
y3=ess.ftf(measVars[np.isin(measVars[:,0],df3),-1])
y3.shape=(y3.shape[0],1)
y2=ess.ftf(measVars[np.isin(measVars[:,0],df2),-1])
y2.shape=(y2.shape[0],1)
y1=ess.ftf(measVars[np.isin(measVars[:,0],df1),-1])
y1.shape=(y1.shape[0],1)

planNum=0
GPDataFile=savePath+"GPData{0}.csv".format(planNum)
GPDataPointers=np.array([df1+df2+df3]).T
fidLevs=np.concatenate((0*np.ones((len(df1),1)),1*np.ones((len(df2),1)),2*np.ones((len(df3),1))),axis=0)
np.savetxt(GPDataFile, np.concatenate((GPDataPointers,fidLevs),axis=1),delimiter=",",header="t,fid",comments="")
Xhs=[Xhf3,Xhf2,Xhf1]
ys=[y3,y2,y1]
Xh_train, Y_train = convert_xy_lists_to_arrays(Xhs,ys)	
n_fids=3
#kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
kernels = [GPy.kern.Matern32(3,ARD=True), GPy.kern.Matern32(3,ARD=True),GPy.kern.Matern32(3,ARD=True)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
lik=GPy.likelihoods.Gaussian()
gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel,likelihood=lik, n_fidelities=n_fids)
gpy_lin_mf_model.param_array[:]=ess.initHypsMF[:]
lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)
if ess.updateGPHyps:
	lin_mf_model.gpy_model.kern.scale.fix([1,1])
	lin_mf_model.gpy_model.kern.rbf.lengthscale.constrain_bounded(0.0001,100)
	lin_mf_model.gpy_model.kern.rbf_1.lengthscale.constrain_bounded(0.0001,100)
	lin_mf_model.gpy_model.kern.rbf_2.lengthscale.constrain_bounded(0.0001,100)
emuHypNames=lin_mf_model.gpy_model.parameter_names()
f=open(savePath+"emuGP.csv","w")
f.write(str(emuHypNames)+"\n")
f.write("rbf_var,rbf_lx,rbf_ly,rbf_lz,rbf1_var,rbf1_lx,rbf1_ly,rbf1_lz,rbf2_var,rbf2_lx,rbf2_ly,rbf2_lz,rho1,rho2,noise,noise1,noise2\n")
f.close()
emuHypVec=lin_mf_model.gpy_model.param_array.copy()
print(emuHypVec,emuHypVec.shape)
f=open(savePath+"emuGP.csv","a")
emuHypVec.shape=(1,emuHypVec.shape[0])
np.savetxt(f,emuHypVec,delimiter=",")
f.close()

USE_SF_IG=1
if USE_SF_IG:
	#kernel=GPy.kern.RBF(input_dim=3,lengthscale=ess.fid2params[1:4],variance=np.sum(lin_mf_model.gpy_model.param_array[[0,4,8]]),ARD=True)
	kernel=GPy.kern.Matern32(input_dim=3,lengthscale=ess.fid2params[1:4],variance=np.sum(lin_mf_model.gpy_model.param_array[[0,4,8]]),ARD=True)
	gp=GPy.models.GPRegression(Xhf1,y1,kernel)
	gp.Gaussian_noise.variance=lin_mf_model.gpy_model.param_array[-1]
#setup RIG planner
agent=ess.agent
agent.CalcCost=agent.calculatePathInfoEmu 
#agent.CalcCost=agent.calculatePathInfoEmuBatch
if USE_SF_IG:
	#agent.CalcCost=agent.calcPathInfoSF
	agent.CalcCost=agent.calcPathInfoSFBatch
	agent.sfgp=gp
#agent=ess.agent
#agent.CalcCost=agent.calculatePathInfoEmu
agent.mfgp=lin_mf_model
print("Max time underwater: ",agent.underWaterTimeLimit)														

BudgetUsed=0
PlannedBudget=0
udot_weights=ess.udot_weights		   
delta_hat=u[2]
k_delta=9#paramsSwim[31]#max vel=paramsSwim[32]
B=ess.B				 					 
BD=ess.BD

nearRad=ess.nearRad
stepSize=ess.stepSize
xstart=np.array([[xhat[0,0]],[xhat[1,0]]])
measFunc=lambda x: 0
planner=IG.Graph(stepSize,B/BD,WS,FS,measFunc,nearRad,agent=agent)
planner.AllowSelfLoops=False
planner.animate = False
planner.ModularCost=False
planner.terminalCond=agent.stopWatch
planner.animateNewEdge = False
planner.debugMode = False	
#planner.maxIter=ess.maxIter
planner.SameNodeDistance=ess.SameNodeDistance
planner.EvaluateCost=agent.getCost
xbee.send(bytes("planning\n",'utf-8'))
startPlanning=time.time()
planner.plan(xstart,Rd=2)
planningTime=time.time()-startPlanning
xbee.send(bytes("done planning\n",'utf-8'))
print("planning time:",planningTime)
agent.stopWatchDuration=ess.planningtime
agent.stopWatchTime=None

trajVars=np.array([[tstart,frameCount,0,xhat[0,0],xhat[1,0],0,0,0,0,0,0]])
delta_d=theta_gd=theta_d=0														  
		 
pathBudget,pathInfo,bnode_idx,bpath_idx=planner.bestPath
PlannedBudget=PlannedBudget+pathBudget									  
print(planner.bestPath)
path=planner.V[bnode_idx].pathList[bpath_idx]
pathPoints=agent.pathToTrajPoints(planner.V,planner.E,path)
allPathPoints=np.hstack((pathPoints,np.ones((pathPoints.shape[0],1))*planNum))																			  
edgeChain2=[planner.E[data[0:2]][data[2]] for data in path]
edgeChain=[]
totalTime=path[-1][3]
for data in path:
	for prim in planner.E[data[0:2]][data[2]][-1]:
		edgeChain.append(prim)
	#totalTime=totalTime+planner.E[data[0:2]][data[2]][4]

f=open(savePath+"plannedTraj{0}.csv".format(planNum),'w')
f.write("x,y,z,t\n")
np.savetxt(f,pathPoints,delimiter=',')
f.close()
f=open(savePath+"bestPath{0}.txt".format(planNum),'w')
f.write(str(planner.bestPath)+"\n")
f.write(str(path)+"\n")
f.write(str(edgeChain)+"\n")
f.write(str(agent.CalcCost)+"\n")
f.write("planning time: "+str(planningTime)+"\n")
f.close()

print(edgeChain)
planner.node_loc_dict(save=True,fname=savePath+"graphNodes{0}.txt".format(planNum))
planner.edge_dict(save=True,fname=savePath+"graphEdges{0}.txt".format(planNum))

savingData=threading.Lock()
useGPS=False
trajTimeStart=time.time()	
t=time.time()-trajTimeStart
toff=0
plannerTrhead=threading.Thread(target=planWatcher)
plannerTrhead.start()
					
tstart=trajTimeStart
tlast_ctrl=tstart
tlast_p_ctrl=tstart
tlast=tstart
u1=u2=0
uc=[0,0,0]
fdt=1/controlRate
p_cnt_last=-1
maxBlue=blue
while running:#time.time()-tstart<60*8:# main loop to collect sensor data and estimate state of miniglider
	t=time.time()-trajTimeStart
	if len(edgeChain)<1 or t>path[-1][3]+toff:
		u1=u2=0
		prim=(None,None,None)
	else:
		p_cnt=min(len(edgeChain)-1,max(0,np.sum(t>pathPoints[:,3])-1))
		prim=edgeChain[p_cnt] if t<path[-1][3]+toff else (None,None,None)
		wypnt=pathPoints[min(p_cnt+1,pathPoints.shape[0]-1)].tolist()
	if p_cnt!=p_cnt_last:
		print(t,p_cnt,prim)
		#print(t,np.sum(t>pathPoints[:,3])-1,pathPoints[:,3])
		p_cnt_last=p_cnt			  
	x_tar,y_tar,z_tar=trajPnt(t,pathPoints)
	xf_tar,yf_tar,zf_tar=trajPnt(t+fdt,pathPoints)#future point for heading
	
	
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
		#running=False
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
	pitchKF,PpitchKF=kalmanPrediction(pitchKF,0,np.array([[1,dt],[0,1]]),0,PpitchKF,0.0175*np.diag([2,3])*dt)
	pitchKF,PpitchKF=kalmanUpdate(pitchKF,PpitchKF,np.array([[pitch],[gy]]),np.diag([1,0]),0.0175*np.diag([1,10]))
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
	elif useGPS and GPSupdate:
		try:
			GPSx,GPSy=(float(GPSData[2]),float(GPSData[3]))
			meas_vec[0]=GPSx
			meas_vec[1]=GPSy
			Phat[0,0]=GPSx
			Phat[1,0]=GPSy
		except:
			meas_vec[0]=aprilState[0,0]
			meas_vec[1]=aprilState[1,0]
			Phat[0,0]=aprilState[0,0]
			Phat[1,0]=aprilState[1,0]
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
		
	useVel_obs=not (dt>.5)*np.isnan(vb_est).any() and ddelta<np.rad2deg(10) and (prim[0]!='Swim' or prim[0]!=None)
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
	
	ess.zErrState,ess.PdepthErrKF=kalmanPrediction(ess.zErrState,0,ess.AdepthErrKF(dt),0,ess.PdepthErrKF,ess.QdepthErrKF*dt)
	ess.zErrState,ess.PdepthErrKF=kalmanUpdate(ess.zErrState,ess.PdepthErrKF,depth-z_tar,ess.HdepthErrKF,ess.RdepthErrKF)
	
	#if int(t)%100==0 and t>int(t) and t-int(t)<.1:# time.time()-tlast_ctrl > 1/controlRate:
	#	print(t,"/",totalTime,prim,path[-1][3]+toff)
		#print(B-BudgetUsed)
	#if (depth>maxDepth):
		#print("\nbelow max depth!")
		#print(t,depth,ess.kMaxDepth*(depth-maxDepth)*(depth>maxDepth))
	#u1=u2=0
	if prim[0]=='FlatDive':
		_,dz,zdot_d=prim
		theta_d=0#np.deg2rad(-2)
		theta_gd=np.pi/2*np.sign(dz)
		if time.time()-tlast_p_ctrl > 1/pitchControlRate:
				u2=ess.massSpdControl(pitch,theta_d,pitchKF[1,0])
				if not nocontrol:
					time.sleep(0.05)
					ch.setMassPos(I2CSock,ch.saturate(100*u[0]+u2/pitchControlRate,0,100))
				tlast_p_ctrl=time.time()			
		if time.time()-tlast_ctrl > 1/controlRate:
			#print(t,"/",totalTime,prim,[x_tar,y_tar,z_tar],[aprilState[0,0],aprilState[1,0],aprilState[2,0]],depth)
			#u1=ess.pumpSpdControl(xhat[2,0],z_tar,wypnt[2],xhat[5,0],zdot_d)
			u1=ess.pumpSpdControl2(depth,ess.zErrState,ess.zErrState[0,0]-wypnt[2])
			if not nocontrol:
				ch.setPumpPos(I2CSock,ch.saturate(100*u[1]+u1/controlRate,0,75))
			tlast_ctrl=time.time()
	
	if prim[0]=='Swim':
		#d,swimSpeed=prim
		if time.time()-tlast_p_ctrl > 1/pitchControlRate:
				theta_d=0.1
				u2=ess.massSpdControl(pitch,theta_d,pitchKF[1,0])
				if not nocontrol:
					time.sleep(0.05)
					ch.setMassPos(I2CSock,ch.saturate(100*u[0]+u2/pitchControlRate,0,100))
				tlast_p_ctrl=time.time()
		if time.time()-tlast_ctrl > 1/controlRate:
			#print(t,"/",totalTime,prim,[x_tar,y_tar,z_tar],[aprilState[0,0],aprilState[1,0],aprilState[2,0]],depth)
			#rho=np.linalg.norm([y_tar-xhat[1,0],x_tar-xhat[0,0]])
			#bearing=np.arctan2(y_tar-xhat[1,0],x_tar-xhat[0,0])
			#bearingNP=np.arctan2(yf_tar-y_tar,xf_tar-x_tar)#bearing from curent point on traj to point at a future time 
			#rho=np.linalg.norm([y_tar-aprilState[1,0],x_tar-aprilState[0,0]]) #testing with actual position
			rho2=np.linalg.norm([wypnt[1]-aprilState[1,0],wypnt[0]-aprilState[0,0]]) #testing with actual position
			#bearing=np.arctan2(y_tar-aprilState[1,0],x_tar-aprilState[0,0]) #testing with actual position	
			bearing=np.arctan2(wypnt[1]-aprilState[1,0],wypnt[0]-aprilState[0,0]) #testing with actual position	
			heading_err=ch.yawCorrection(yaw,bearing,np.pi)
			zdot_d=0
			bias=ch.saturate(3*np.rad2deg(heading_err),-90,90)
			amp=np.rad2deg(agent.tailAmp) if rho2>0.5 else 100*rho2*agent.tailAmp/50*(np.cos(heading_err)>0)
			amp=ch.saturate(amp,0,50)
			
			tail.bias=bias
			tail.amp=amp
			tail.freq=agent.tailFreq
			
			#u1=ess.pumpSpdControl(xhat[2,0],z_tar,wypnt[2],xhat[5,0],zdot_d)
			u1=ess.pumpSpdControl2(depth,ess.zErrState,ess.zErrState[0,0]-wypnt[2])
			if not nocontrol:
				ch.setPumpPos(I2CSock,ch.saturate(100*u[1]+u1/controlRate,0,75))
			tlast_ctrl=time.time()
	else:
		tail.amp=0
	if prim[0]=='Spiral':
		_,dz,delta_d,zdot_d=prim
		if time.time()-tlast_ctrl > 1/controlRate:
			#print(t,"/",totalTime,prim,[x_tar,y_tar,z_tar],[aprilState[0,0],aprilState[1,0],aprilState[2,0]],depth)
			theta_d=pitch
			#u1=ess.pumpSpdControl(xhat[2,0],z_tar,wypnt[2],xhat[5,0],zdot_d)
			u1=ess.pumpSpdControl2(depth,ess.zErrState,ess.zErrState[0,0]-wypnt[2])
			u2=ess.massSpdControl(pitch,theta_d,0)
			if not nocontrol:
				ch.setPumpPos(I2CSock,ch.saturate(100*u[1]+u1/controlRate,0,75))
				time.sleep(0.05)
				if useGPS and dz<0:
					ch.setMassPos(I2CSock,ess.masStart)
					time.sleep(0.05)
				elif (dz>0.1 or dz<0):
					ch.setMassPos(I2CSock,35 if dz>0 else 60)
					time.sleep(0.05)
			tail.bias=np.rad2deg(delta_d)
			tlast_ctrl=time.time()
			
		
	if prim[0]=='Glide':
		_,theta_gd,dz,zdot_d=prim
		if time.time()-tlast_p_ctrl > 1/pitchControlRate:
				theta_d=-theta_gd if abs(theta_gd)<np.deg2rad(45) else -np.pi/2*np.sign(theta_gd)+theta_gd	
				if useGPS and dz<0:
					theta_d=0
				u2=ess.massSpdControl(pitch,theta_d,pitchKF[1,0])
				if not nocontrol:
					time.sleep(0.05)
					ch.setMassPos(I2CSock,ch.saturate(100*u[0]+u2/pitchControlRate,0,100))
				tlast_p_ctrl=time.time()
		if time.time()-tlast_ctrl > 1/controlRate:
			#xtemp,ytemp=(hat[0,0],Phat[1,0])
			xtemp,ytemp=(aprilState[0,0],aprilState[1,0])#testing with actual position
			
			bearing=np.arctan2(wypnt[1]-ytemp,wypnt[0]-xtemp) 
			
			#u1=ess.pumpSpdControl(xhat[2,0],z_tar,wypnt[2],xhat[5,0],zdot_d)
			u1=ess.pumpSpdControl2(depth,ess.zErrState,ess.zErrState[0,0]-wypnt[2])
			#u2=ess.massSpdControl(pitch,theta_d,pitchKF[1,0])
			if not nocontrol:
				ch.setPumpPos(I2CSock,ch.saturate(100*u[1]+u1/controlRate,0,75))
				time.sleep(0.01)
			tail.bias=ch.saturate(np.rad2deg(ch.yawCorrection(yaw,bearing,np.pi)),tail.bias-ess.maxBiasRate*dt,tail.bias+ess.maxBiasRate*dt)#np.rad2deg(uc[2])								
			#tail.bias=np.rad2deg(ch.yawCorrection(yaw,bearing,np.pi))#np.rad2deg(uc[2])																  
	if prim[0]==None and agent.stopWatchTime!=None:
		if depth>ess.atSurface*.5 and time.time()-tlast_ctrl > 10/controlRate and not nocontrol:
			tlast_ctrl=time.time()	
			ch.setPumpPos(I2CSock,ch.saturate(100*u[1]+3,0,75))
			time.sleep(0.05)
			ch.setMassPos(I2CSock,ch.saturate(ess.massStart+8*(100*u[1]/ess.pumpStart-1),0,100))
			
	savingData.acquire()	
	estimationVars=np.append(estimationVars,[[tlast,frameCount,Phat[0,0],Phat[1,0],Phat[2,0],xhat[0,0],xhat[1,0],xhat[2,0],xhat[3,0],xhat[4,0],xhat[5,0],Pxhat[0,0],Pxhat[1,1],Pxhat[2,2],Pxhat[3,3],Pxhat[4,4],Pxhat[5,5],vb_est[0,0],vb_est[1,0],vb_est[2,0],dvb[0,0],dvb[1,0],dvb[2,0],BudgetUsed,PlannedBudget,agent.stopWatchTime!=None]],axis=0) 
	measVars=np.append(measVars,[[tlast,frameCount,u[0], u[1], u[2], depth, roll, pitch, yaw,yaw2,r,pitchKF[1,0], gx, gy, gz,ax ,ay, az,battV, trgb, red,green,blue]],axis=0) 
	savingData.release()
	cntrlVars=np.append(cntrlVars,[[tlast,frameCount,u2,u1,tail.bias,tail.amp,tail.freq,uc[0],uc[1],uc[2],dmass,dpump,delta_hat,ddelta]],axis=0) 
	tempSpriral=delta_d if prim[0]=='Spiral' else np.nan
	tempTheta_d=np.nan if prim[0]==None else theta_d 
	tempTheta_gd=theta_gd if prim[0]=='Glide' or prim[0]=='FlatDive' else np.nan 
	trajVars=np.append(trajVars,[[tlast,frameCount,t,x_tar,y_tar,z_tar,wypnt[0],wypnt[1],tempTheta_d,theta_gd,tempSpriral]],axis=0)
																															
			 
	#if 60*5*agent.timeEnergy-BudgetUsed*0<0:
	#if B-BudgetUsed<0:
	#	running=False
		
		#save data	


																									 
	if max(estimationVars.shape)>1000:
		savingData.acquire()
		f=open(savePath+"estimates.csv",'a')
		np.savetxt(f,estimationVars,delimiter=',')
		f.close()
		estimationVars=np.empty((0,estimationVars.shape[1]))
		savingData.release()
	if max(cntrlVars.shape)>1000:
		f=open(savePath+"control.csv",'a')
		np.savetxt(f,cntrlVars,delimiter=',')
		f.close()
		cntrlVars=np.empty((0,cntrlVars.shape[1]))
	if max(trajVars.shape)>1000:
		f=open(savePath+"trajInfo.csv",'a')
		np.savetxt(f,trajVars,delimiter=',')
		f.close()
		trajVars=np.empty((0,trajVars.shape[1]))
	if max(measVars.shape)>1000:
		savingData.acquire()
		f=open(savePath+"measurements.csv",'a')
		np.savetxt(f,measVars,delimiter=',')
		f.close()
		measVars=np.empty((0,measVars.shape[1]))
		savingData.release()
	time.sleep(1/1000)
	#if time.time()-trajTimeStart>totalTime:
	#	break
		
running=False
tail.stop()

savingData.acquire()
f=open(savePath+"measurements.csv",'a')
np.savetxt(f,measVars,delimiter=',')
f.close()
f=open(savePath+"estimates.csv",'a')
np.savetxt(f,estimationVars,delimiter=',')
f.close()  
f=open(savePath+"control.csv",'a')
np.savetxt(f,cntrlVars,delimiter=',')
f.close()  
f=open(savePath+"trajInfo.csv",'a')
np.savetxt(f,trajVars,delimiter=',')
f.close()  
f=open(savePath+"plannedTrajAll.csv".format(planNum),'w')
f.write("x,y,z,t,planNum\n")
np.savetxt(f,allPathPoints,delimiter=',')
f.close()		 
savingData.release()
			 
LEDSock.send("random".encode('utf-8'))
ch.setPumpSpd(I2CSock,99)
time.sleep(0.05)
ch.setMassPos(I2CSock,45)	
time.sleep(1)
LEDSock.send("off".encode('utf-8'))
xbee.send(bytes("done\n",'utf-8'))
xbee.send(bytes("stopUpdates",'utf-8'))
print("Done...")
#msg to send from control computer to stop video: OBTTC,STOP

estData=np.loadtxt(savePath+"estimates.csv",skiprows=1,delimiter=",")
measData=np.loadtxt(savePath+"measurements.csv",skiprows=1,delimiter=",")
print("Blue stats: mean=",np.mean(measData[:,-1]),", max=",np.max(measData[:,-1]),", min=",np.min(measData[:,-1]))
#extract relevant stuff for GP and train
Xhf3=estData[np.isin(estData[:,0],df3),5:8]
Xhf2=estData[np.isin(estData[:,0],df2),5:8]
Xhf1=estData[np.isin(estData[:,0],df1),5:8]
y3=ess.ftf(measData[np.isin(measData[:,0],df3),-1])
y3.shape=(y3.shape[0],1)
y2=ess.ftf(measData[np.isin(measData[:,0],df2),-1])
y2.shape=(y2.shape[0],1)
y1=ess.ftf(measData[np.isin(measData[:,0],df1),-1])
y1.shape=(y1.shape[0],1)
GPDataFile=savePath+"GPData{0}.csv".format(planNum)
GPDataPointers=np.array([df1+df2+df3]).T
fidLevs=np.concatenate((0*np.ones((len(df1),1)),1*np.ones((len(df2),1)),2*np.ones((len(df3),1))),axis=0)
np.savetxt(GPDataFile, np.concatenate((GPDataPointers,fidLevs),axis=1),delimiter=",",header="t,fid",comments="")
Xhs=[Xhf3,Xhf2,Xhf1]
ys=[y3,y2,y1]
Xh_train, Y_train = convert_xy_lists_to_arrays(Xhs,ys)	

#lin_mf_model.optimize()
last_params=lin_mf_model.gpy_model.param_array
if ess.updateGPHyps:
	kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
	lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
	lik=GPy.likelihoods.Gaussian()
	gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel,likelihood=lik, n_fidelities=n_fids)
	lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)
	lin_mf_model.gpy_model.kern.scale.fix([1,1])
	lin_mf_model.gpy_model.kern.rbf.lengthscale.constrain_bounded(0.0001,100)
	lin_mf_model.gpy_model.kern.rbf_1.lengthscale.constrain_bounded(0.0001,100)
	lin_mf_model.gpy_model.kern.rbf_2.lengthscale.constrain_bounded(0.0001,100)
	if not np.any(np.isnan(last_params)):
		lin_mf_model.gpy_model.param_array[:]=last_params[:] 
	lin_mf_model.optimize()
else:
	lin_mf_model.set_data(Xh_train,Y_train)
emuHypVec=lin_mf_model.gpy_model.param_array.copy()
print(emuHypVec,emuHypVec.shape)
f=open(savePath+"emuGP.csv","a")
emuHypVec.shape=(1,emuHypVec.shape[0])
np.savetxt(f,emuHypVec,delimiter=",")
f.close()	

testPoints = ess.testPoints
mu,sig=lin_mf_model.predict(np.hstack((testPoints,2*np.ones((testPoints.shape[0],1)))))
np.savetxt(savePath+"resultsSF.csv",np.concatenate((testPoints,mu,sig),axis=1),delimiter=",",header=" x,y,z,gpMean,gpVar",comments="")
print("GP Trained. End of script.")
