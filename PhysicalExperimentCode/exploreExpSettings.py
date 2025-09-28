import numpy as np
import ergodicKLDivergence as ekld
import GraceRIGV3 as IG
import backsteppingConfig as cfg
import aprilTagLocations as atl
import controllerHelper as ch

def getEID(gp,WS,mD,testSet=None,emu=False,alpha=0.2):	
	if type(testSet)==type(None):
		#specs3D=[[WS[0,0],WS[0,1],20],[WS[1,0],WS[1,1],13],[0,mD,5]]
		#dim = len(specs3D)
		#grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
		#ss3D = np.array([grid3D[i].ravel() for i in range(dim)]).T
		ss3D = ERGfieldGrid
	else:
		ss3D=testSet
	if emu:
		mu,sig=gp.predict(np.hstack((ss3D,np.ones((ss3D.shape[0],1))*2)))
		prior_sig=np.sum(gp.gpy_model.param_array[[0,4,8,-1]]) #variance with no data
		
	else:
		mu,sig=gp.predict(ss3D)
		prior_sig=gp.kern.variance[0]+gp.Gaussian_noise.variance[0] #variance with no data
	sig[sig<0]=prior_sig
	if auto:
		alpha=1-np.mean(sig)/prior_sig#alpha=1-np.max(sig)/prior_sig
		#alpha=1-0.5*(np.mean(sig)+np.max(sig))/prior_sig#alpha=1-np.max(sig)/prior_sig
	fauxUCB=alpha*mu+(1-alpha)*np.sqrt(sig)
	EID=ekld.softmax(fauxUCB)
	return EID,ss3D
	
def pumpSpdControl(depth,z_tar,zwpnt,zdot,zdot_d):
	dkp,dkd=linearDepthGains
	ddz=abs(depth-z_tar)>0.1 and np.sign(depth-z_tar)== np.sign(depth-zwpnt)#0 if in depth dead zone
	sdz=abs(zdot-zdot_d*(abs(depth-zwpnt)>0.1))>0.005#0 if in speed dead zone
	
	u1=ch.saturate(dkd*(zdot-zdot_d)*sdz,-100,100) \
		+ch.saturate(dkp*(depth-z_tar)*ddz,-100,100) \
		+kMaxDepth*(depth-maxDepth)*((depth+.001)>maxDepth)
	u1=ch.saturate(u1,-100,100)
	return u1
	
def pumpSpdControl2(depth,e_state,ewpnt):
	dkp,dkd,ddkd,dddkd=linearDepthGains2
	e_tar,de_tar,dde_tar,ddd_etar=e_state.flatten()
	ddz=abs(e_tar)>0.075 and np.sign(e_tar)== np.sign(ewpnt)#0 if in depth dead zone
	sdz=abs(de_tar*(abs(ewpnt)>0.1))>0.005#0 if in speed dead zone
	#u1=ch.saturate(dkd*(de_tar)*sdz,-100,100) \
	#	+ch.saturate(dkp*(e_tar)*ddz,-100,100) \
	#	+kMaxDepth*(depth-maxDepth)*((depth+.001)>maxDepth)	
	#u1=np.dot(linearDepthGains2,e_state*np.array([sdz,ddz,,1,1]).T)+kMaxDepth*(depth-maxDepth)*((depth+.001)>maxDepth)
	u1=np.dot(linearDepthGains2,e_state)+kMaxDepth*(depth-maxDepth)*((depth+.001)>maxDepth)
	u1=ch.saturate(u1[0],-100,100)
	return u1
	
def massSpdControl(pitch,theta_d,pitchVel):
	pkp,pkd=linearPitchGainsp
	e=theta_d-pitch
	pdz=1#abs(e)>np.deg2rad(3)# pitch dead zone
	u2=ch.saturate(pkd*(-pitchVel)*pdz,-100,100) \
		+ch.saturate(pkp*e*pdz,-100,100) 
		#+ch.saturate(pkp*(1+1*(abs(e)<np.deg2rad(9)))*(e)*pdz,-100,100) 
	
	#u2=ch.saturate(u2,-100,100) if abs(u2/pitchControlRate)>minDist else minDist*pitchControlRate*np.sign(u2)*(abs(u2)>0.2)
	u2=ch.saturate(u2,-100,100) #if abs(u2/pitchControlRate)>minDist else 0.25*pitchControlRate*u2
	return u2
	
minDist=0.5	

#np.random.seed(0)
auto=0
nocontrol=False
updateGPHyps=False
blueThresh=.95
sig_var=3.378#4.2
lenscale=[.1678,.1792,.3618]#[.09,.09,.228]
measNoise=1e-8#.02
initHyps=np.array([sig_var]+lenscale+[measNoise])
fid1params,fid2params,fid3params,scaleParams,mfMeasNoise=([6.6895,.3872,.3808,.4076],[1.9063,.1938,.1868,.2204],[3.72e-8,4.78,3.65,1.8],[1,1],0.1156)
#fid1params,fid2params,fid3params,scaleParams,mfMeasNoise=([.4376,7.04e-3,1.45e-2,1.01e-2],[3.09,.594,.6446,.626],[1.343,.591,.495,.255],[1,1],1.69e-3)
initHypsMF=np.array(fid1params+fid2params+fid3params+scaleParams+[mfMeasNoise])
#act speed controller settins
linearDepthGains=(100,3000)#dkp,dkd->linear depth controller gains
linearDepthGains2=(100,3000,20,3)#dkp,dkd,ddkd,dddkd->linear depth controller gains
kMaxDepth=500000#extra term on depth control to keep it from going too far over maxDepth
#linearPitchGainsp=(10,3)#kp,kd->linear pitch controller gains
linearPitchGainsp=(5,0.5)#kp,kd->linear pitch controller gains
#pitchPIDGains=(35,0.005,0.02)#kp,ki,kd	
#pitchPIDGains=(35,0.8,20)#kp,ki,kd	
pitchPIDGains=(35,0.8,200)#kp,ki,kd	
#act pos controller settings
pumpStart=55
massStart=46
controlRate=10#hz
pitchControlRate=4#hz
maxBiasRate=100

#AprilTag Kalman Filter 
Papril=0.5*np.eye(8)
Qapril=np.diag([0.25 ,0.25 ,0.25,np.deg2rad(3),0.05,0.05,0.05,np.deg2rad(1.5)])
Rmeas1april=0.02
Rmeas2april=[.75,0.75,0.75,np.deg2rad(10)**2]
Aapril=np.eye(8)
rdtapril=[0,1,2]
cdtapril=[4,5,6]
Bapril=np.array([[0],[0],[0],[1],[0],[0],[0],[0]])
GPSxynoise=.15**2
GPSyawnoise=np.deg2rad(5)**2
velVarMult=3
#setup kalman filter for position estimate. state: x,y,z,z_dot
Pxhat=.1*np.eye(4)
Qxhat=np.diag([0.005,0.005,0.005,0.1])
Axhat=lambda dt:np.eye(4)+np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])*dt
Bxhat=np.eye(4)
Bxhat[3,3]=0
Rxhat=np.diag([.1,.1,.05])

#setup alternate kalman filter for position estimate. state: x,y,z,x_dot,y_dot,z_dot
Pxhat2=1*np.eye(6)
Qxhat2=np.diag([0.001,0.001,0.001,0.01,0.01,.01])
damping=-0.01
Axhat2=lambda dt:np.eye(6)+np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,damping,0,0],[0,0,0,0,damping,0],[0,0,0,0,0,damping]])*dt
Bxhat2=0
Rxhat2=np.diag([.1,.1,.05,0.25,0.25,0.25,0.35,0.35,0.35])#meas:x,y,z,dx_april,dy_april,_dz_april,dx_nlobs,dy_nl_obs,dz_nl_obs

#input speed kf. state: mass position,pump position, mass speed, pump speed
Pinp=.1*np.eye(4)
Qinp=np.diag([0.05,0.05,0.05,0.05])
Ainp=lambda dt:np.eye(4)+np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])*dt
Rinp=np.diag([.1,.1])/100
Hinp=np.array([[1,0,0,0],[0,1,0,0]])

#pitch kalman filter. state: pitch, pitch_dot
Ppitch=.1*np.eye(2)
Qpitch=np.diag(np.deg2rad([2**2,5**2]))
Apitch=lambda dt:np.eye(2)+np.array([[0,1],[0,0]])*dt
Bpitch=0
Rpitch=np.diag([np.deg2rad(2)])

#depth error kalman filter. state: e, de, ,dde, ddde
zErrState=np.array([[0,0,0,0]]).T
PdepthErrKF=.1*np.eye(4)
QdepthErrKF=np.diag([.1,.1,.1,.1])
AdepthErrKF=lambda dt:np.eye(4)+np.eye(4,k=1)*dt+np.eye(4,k=2)/2*dt**2+np.eye(4,k=3)/6*dt**3
RdepthErrKF=np.diag([0.05])
HdepthErrKF=np.array([[1,0,0,0]])

#function to interpolate (linear) 3D path
trajPnt= lambda x,y: np.array([np.interp(x,y[:,3],y[:,0]),np.interp(x,y[:,3],y[:,1]),np.interp(x,y[:,3],y[:,2])])

#field settings	
feetToMeter=1/3.28
WS=np.array([[3,12],[1.5,6]])*feetToMeter
maxDepth=.65
fidlevels=(2*(min(np.diff(WS))*np.array([0.05,0.15,.25]))).tolist()
ftf=lambda x: np.log(x+1)#field transform for output data 

#eval for batch Ergodic
specs3D=[[WS[0,0],WS[0,1],21],[WS[1,0],WS[1,1],11],[0,maxDepth,5]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
ERGfieldGrid = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T
#ecal for batch infogain
specs3D=[[WS[0,0],WS[0,1],10],[WS[1,0],WS[1,1],6],[0,maxDepth,5]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
IGfieldGrid = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T
#evaluation settings
#specs3D=[[WS[0,0],WS[0,1],21],[WS[1,0],WS[1,1],21],[0,1,11]]
specs3D=[[0,15*feetToMeter,31],[0,10*feetToMeter,31],[0,1,11]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
testPoints = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T
'''
tx,ty,tz=np.meshgrid(np.linspace(WS[0,0],WS[0,1],20),np.linspace(WS[1,0],WS[1,1],10),np.linspace(0,maxDepth,20))
tx,ty,tz=(tx.ravel('F'),ty.ravel('F'),tz.ravel('F'))
tx.shape=(tx.shape[0],1)
ty.shape=(ty.shape[0],1)
tz.shape=(tz.shape[0],1)
X_star1 = np.concatenate((tx,ty,tz),axis=1)
muv2,sigv2=gp.predict(X_star1)
mu2v2,sig2v2=lin_mf_model.predict(np.concatenate((X_star1,np.zeros((X_star1.shape[0],1))),axis=1))
np.savetxt("Data/resultsv2.csv",np.concatenate((tx,ty,tz,f(X_star1),muv2,sigv2,mu2v2,sig2v2),axis=1),delimiter=",",header=" x,y,z,trueField,gpMean,gpVar,emuMean,emuVar")
'''

atSurface=0.15
savePath="Data/"#cfg.BSsavePath
params=np.loadtxt(cfg.paramPath,delimiter=',')
paramsFlat=np.loadtxt(cfg.paramPath,delimiter=',')
paramsSwim=np.loadtxt("calibrationData/params_swim.model",delimiter=',')



#agent settings 
goalVar=1**2
trajCount=3
measRate=1/2
SurfaceBySpiral=False
swimSpeed=.05
spiralSpeed=.015
vertGlideSpeed=.015
flatDiveSpeed=.015

FlatDiveEnergy=1
GlideEnergy=1.5
tailEnergyScale=.2
timeEnergy=0.005
#agent.SwimEnergy(1,agent.tailAmp,agent.tailFreq)*agent.tailEnergyScale
#udot_weights=np.array([.5,9,.04,1])#rp1,m0,delta,time
#udot_weights=np.array([.05,.1,.004,1])#rp1,m0,delta,time
udot_weights=np.array([1,1,1,1])#rp1,m0,delta,time
udot_weightsSwim=udot_weights*np.array([1,1,100,1])#rp1,m0,delta,time

planningtime=45
initialPlanningTime=45
agent=IG.GraceAgent()
agent.fieldGrid=IGfieldGrid
agent.stopWatchDuration=initialPlanningTime
agent.tailFreq=1#0.75
agent.tailAmp=np.deg2rad(25)#45
#agent.CalcCost=agent.calcPathInfoSF
#agent.sfgp=gp
#agent.CalcCost=agent.calcPathInfo 
#agent.mfgp=gp
#agent.CalcCost=agent.calcPathErgodicity# 
#agent.EID,agent.fieldGrid=ess.getEID(gp,WS,maxDepth)
#agent.ergSigma=0.1*np.eye(3)
#legTypes=["Spiral","Glide","Swim","FlatDive"]
agent.legProbs=[0,1/3,1/3,1/3]
#agent.legProbs=[0/3,1,0,0]
#agent.legProbs=[.5/3,1/3,1/3,.5/3]
#agent.legProbs=[.5/3,0/3,2/3,.5/3]
#agent.legProbs=[0/3,0/3,3/3,0/3]
agent.fidLevs=fidlevels
agent.trajCount=trajCount
agent.measRate=measRate
agent.maxDepth=maxDepth
agent.SurfaceBySpiral=SurfaceBySpiral
agent.swimSpeed=swimSpeed
agent.spiralSpeed=spiralSpeed
agent.vertGlideSpeed=vertGlideSpeed
agent.flatDiveSpeed=flatDiveSpeed
# for budget calculation
agent.FlatDiveEnergy=FlatDiveEnergy
agent.GlideEnergy=GlideEnergy
agent.tailEnergyScale=tailEnergyScale
agent.timeEnergy=timeEnergy
agent.varianceRate=Qxhat[0,0]
agent.underWaterTimeLimit=(goalVar)/Qxhat[0,0]


#planner settings
B=80#120
BD=4#6#BudgetDivisor
SameNodeDistance=.1
maxIter=100
Rd=2
nearRad=.125
stepSize=2
initIters=100
maxIter=100
pathEpsilon=1.25
planningCost=1

xmax,xmin,ymax,ymin=[max(atl.tankPoses[:,1]),min(atl.tankPoses[:,1]),max(atl.tankPoses[:,2]),min(atl.tankPoses[:,2])]#xmin,xmax,ymin,ymax
	
#setup Data File Headers
np.savetxt(savePath+"model_params.csv",params,delimiter=',')

header="t,frame,x,y,z,xkf,ykf,zkf,dxkf,dykf,dzkf,sig_xkf,sig_ykf,sig_zkf,sig_dxkf,sig_dykf,sig_dzkf,v1,v2,v3,dv1,dv2,dv3,BudgetUsed,PlannedBudget,planning"
f=open(savePath+"estimates.csv",'w')
f.write(header+'\n')
f.close()

header="t,frame,mass%,pump%,delta,depth,roll,pitch,yaw,yaw2,yawRateIMU,pitchRate,gyrox,gyroy,gyroz,ax,ay,az,battV,trgb,red,green,blue"
f=open(savePath+"measurements.csv",'w')
f.write(header+'\n')
f.close()

header="t,frame,mass_dot,pump_dot,bias,amp,freq,rp1_glide,m0_glide,delta_glide,dmass_kf,dpump_kf,delta_hat,ddelta_hat"
f=open(savePath+"control.csv",'w')
f.write(header+'\n')
f.close()

header="t,frame,t_traj,x_tar,y_tar,z_tar,xw,yw,theta_d,theta_g,spiral_ang"
f=open(savePath+"trajInfo.csv",'w')
f.write(header+'\n')
f.close()

f=open(savePath+"plannedTrajAll.csv",'w')
f.write("x,y,z,t,planNum\n")
f.close()

import __main__
f=open(savePath+"__"+__main__.__file__.strip(".py"),"w")
f.write("linearDepthGains:"+str(linearDepthGains2)+"\n")
f.write("linearPitchGains:"+str(linearPitchGainsp)+"\n")
f.close()
