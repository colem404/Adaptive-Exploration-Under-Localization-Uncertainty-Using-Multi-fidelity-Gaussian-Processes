import numpy as np
import GPy
import scipy 
import matplotlib.pyplot as plt
import GraceObservers 
from GraceObservers import kalmanUpdate,kalmanPrediction
import GraceRIGV3 as IG
import time
import controllerHelper as ch
import ergodicKLDivergence as ekld
import exploreSimSettings as ess
	

def bicycle3D():
	pass

def integrator(f,x,h):
	k1=f(x)
	k2=f(x+k1*h/2)
	k3=f(x+k2*h/2)
	k4=f(x+k3*h)
	xnew=x+h/6*(k1+2*k2+2*k3+k4)
	return xnew

def graceSimple(x,u):
	xp,yp,zp,pitch,yaw,v1,v3=x.flatten()
	g,rho=(9.81,1000)
	alpha=np.arctan2(v3,v1)
	L=0.5*rho*V**2*np.sin(alpha)**2
	D=0.5*rho*V**2
	V=np.sqrt(v1*2+v3**2)
	w1,w2,a,thrust=u
	dx=V*np.cos(yaw)*np.cos(pitch-alpha)
	dy=V*np.sin(yaw)*np.cos(pitch-alpha)
	dz=V*np.sin(pitch-alpha)
	dv1=g*np.sin(pitch)*a+thrus+L*np.sin(alpha)-D*np.cos(alpha)
	dv3=-g*np.cos(pitch)*a+L*np.sin(alpha)-D*np.sin(alpha)
	dyaw=w1
	dpitch=w2
	return np.array([[dx],[dy],[dz],[dpitch],[dyaw],[dV]])

def singleIntegrator3D(x,u):
	xp,yp,zp=x.flatten()
	vx,vy,vz=u
	dx=vx
	dy=vy
	dz=vz
	return np.array([[dx],[dy],[dz]])
	
def Unicycle3D(x,u,alpha=0):
	xp,yp,zp,pitch,yaw,V=x.flatten()
	pitch=np.arcsin(np.sin(pitch))
	w1,w2,a=u
	dx=V*np.cos(yaw)*np.cos(pitch-alpha)
	dy=V*np.sin(yaw)*np.cos(pitch-alpha)
	dz=V*np.sin(pitch-alpha)
	dV=-(0.5+np.sin(alpha)**2)*V+a
	dyaw=w1
	dpitch=w2
	return np.array([[dx],[dy],[dz],[dpitch],[dyaw],[dV]])
	
savePath="Data/"
#function to interpolate (linear) 3D path
#function to interpolate (linear) 3D path
trajPnt= ess.trajPnt
f=open(savePath+"fieldSettings.txt","w")
f.write("Type: WRBFField\n")
f.write("L,s,w: "+str((ess.L,ess.s,ess.w))+"\n")
f.write("sources: "+str(ess.p)+"\n")
f.close()
	
#setup measurment field
WS=ess.WS
maxDepth=ess.maxDepth
FS=WS
p=ess.p
measFunc=ess.measFunc



specs3D=[[WS[0,0],WS[0,1],8],[WS[1,0],WS[1,1],16],[0,maxDepth,8]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
Xhs = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T
ys=ess.f(Xhs)+ess.measNois*np.random.normal(size=(Xhs.shape[0],1))
print(Xhs.shape,ys.shape)
np.savetxt(savePath+"trainSF.csv",np.concatenate((Xhs,ys),axis=1),delimiter=",",header=" x,y,z,fieldMeas",comments="")
#GP model
mean=None#GPy.mappings.Constant(3,1)
kernel=GPy.kern.RBF(input_dim=3,variance=1,lengthscale=1,ARD=True)
gp=GPy.models.GPRegression(Xhs,ys,kernel,mean_function=mean)
HypNames=gp.parameter_names()
f=open(savePath+"GPySFGP.txt","w")
f.write(str(HypNames)+"\n")
f.write("rbf_var,rbf_lx,rbf_ly,rbf_lz,noise\n")
f.close()
gp.optimize()
HypVec=gp.param_array.copy()
print(HypVec,HypVec.shape)
f=open(savePath+"GPySFGP.txt","a")
HypVec.shape=(1,HypVec.shape[0])
np.savetxt(f,HypVec,delimiter=",")
f.close()
		
testPoints=ess.testPoints
mu,sig=gp.predict(testPoints)
fTrue=ess.f(testPoints)
np.savetxt(savePath+"resultsSF.csv",np.concatenate((testPoints,fTrue,mu,sig),axis=1),delimiter=",",header=" x,y,z,trueField,gpMean,gpVar",comments="")
print("RMSE:",np.sqrt(np.mean((mu-fTrue)**2)))