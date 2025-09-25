# code takes a .csv file with a trajectory (t,x,y,z) and estimates it using a kalman filter.
import numpy as np
from GraceObservers import kalmanUpdate,kalmanPrediction
import exploreSimSettings as ess

TrajName="T"+str(ess.seed)+"_"+str(ess.vmn)
file_path='..\V4\Data\simpMF\seed0\mfgpSimSimp.csv'
with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')
        print(headers)
        trajData = np.loadtxt(f, delimiter=',')
	
savePath="Data/TrajectoriesAndEstimates/"
#function to interpolate (linear) 3D path
trajPnt= ess.trajPnt
f=open(savePath+TrajName+"Settings.txt","w")
f.write("Groundtruth Origin: "+file_path+"\n")
f.write("Random Seed: "+str(ess.seed)+"\n")
f.write("Meas Noise:\n "+str(ess.kfMeasNoise)+"\n")
	
#setup measurment field
WS=ess.WS
FS=WS
p=ess.p
													  
xnow=trajData[0,[headers.index("x"),headers.index("y"),headers.index("z")]]
u=[0,0,0]#vx,vy,vz
t=0
#setup kalman filter for position 
xhat=np.array([xnow.flatten().tolist()[0:3]+[0,0,0]]).T	#x,y,z,vx,vy,vz
Pxhat=ess.Pxhat2
Qxhat=ess.Qxhat2
Axhat=ess.Axhat2
Bxhat=ess.Bxhat2
Rxhat=ess.Rxhat2
#set up dataset

f.write("KF A("+str(ess.dt)+") Matrix:\n "+str(Axhat(ess.dt))+"\n")
f.write("KF B Matrix:\n "+str(Bxhat)+"\n")
f.write("KF Pinit Matrix:\n "+str(Pxhat)+"\n")
f.write("KF Q Matrix:\n "+str(Qxhat)+"\n")
f.write("KF R Matrix:\n "+str(Rxhat)+"\n")
f.close()

dataFile=open(savePath+TrajName+".csv",'w')
print(savePath+TrajName+".csv")
dataFile.write("t,x,y,z,xh,yh,zh,sigx,sigy,sigz,xe,ye,ze\n")

cnt=trajData.shape[0]
lastSampleTime=trajData[0,headers.index("t")]
for j in range(1, cnt):	
	xtemp=trajData[j-1,[headers.index("x"),headers.index("y"),headers.index("z")]].tolist()
	t=trajData[j-1,headers.index("t")]
	dt=trajData[j,headers.index("t")]-trajData[j-1,headers.index("t")]
	if t>3600:
		break					
	#estimate state
	#kalman filter stuff
	u[0]=(trajData[j,headers.index("x")]-trajData[j-1,headers.index("x")])/(trajData[j,headers.index("t")]-trajData[j-1,headers.index("t")])
	u[1]=(trajData[j,headers.index("y")]-trajData[j-1,headers.index("y")])/(trajData[j,headers.index("t")]-trajData[j-1,headers.index("t")])
	u[2]=(trajData[j,headers.index("z")]-trajData[j-1,headers.index("z")])/(trajData[j,headers.index("t")]-trajData[j-1,headers.index("t")])
	gamGPS= xtemp[2]<=ess.atSurface
	Hxhat=np.diag([gamGPS,gamGPS,1,1,1,1])
	kfmeas=np.array([[xtemp[0]],[xtemp[1]],[xtemp[2]],[u[0]],[u[1]],[u[2]]])
	kfmeas=kfmeas+ess.kfMeasNoise*np.random.normal(size=kfmeas.shape)
	
	xhat,Pxhat=kalmanPrediction(xhat,0,Axhat(dt),Bxhat,Pxhat,Qxhat*dt)
	xhat,Pxhat=kalmanUpdate(xhat,Pxhat,kfmeas,Hxhat,Rxhat)
	sigx,sigy,sigz,_,_,_=np.diag(Pxhat).tolist()
	xe,ye,ze=[xtemp[0]-xhat[0,0],xtemp[1]-xhat[1,0],xtemp[2]-xhat[2,0]]
	tempDataArr=np.array([[t]+xtemp+[xhat[0,0],xhat[1,0],xhat[2,0],sigx,sigy,sigz,xe,ye,ze]])
	np.savetxt(dataFile,tempDataArr,delimiter=",")
	

		


dataFile.close()

