import numpy as np

def traj(name,t):
	if name=="circle":
		w=np.pi*2
		f1,f2,f3=(1/150,1/150,1/150)
		a1,a2,a3,a4=(1,1,0.3,20)
		return (a1*np.sin(w*f1*t),a2*np.sin(w*f2*t),0.3+a3*np.sin(w*f3*t),np.deg2rad(a4)*np.sign(np.sin(w*f3*(t+.1))-np.sin(w*f3*t)))
	if name=="line":
		w=np.pi*2
		f=1/90
		a=25
		return (-1+0.015*t,0.0,0.35-0.2*np.cos(w*f*t),np.deg2rad(a)*np.sign(np.cos(w*f*(t+.1))-np.cos(w*f*t)))
	if name=="line2":
		w=np.pi*2
		f=1/75
		a=35
		return (-1+t*0.012,-1+0.01*t,0.35-0.2*np.cos(w*f*t),-np.deg2rad(a)*np.sin(w*f*t))
	if name=='pringle':
		f=1/60
		w=2*np.pi
		a=.5
		f2=.5*f
		x=.5*a*np.sin(w*f2*t)
		y= a*np.cos(w*f2*t)
		z= 0.4-0.1*np.cos(w*f*t)
		pitch=-np.deg2rad(20)*np.sin(w*f*t)
		return (x,y,z,pitch)
	if name=='ellipse':
		f,f2=(1/90,1/270)
		w=2*pi
		a1,a2=(1,1)
		x=a1*np.cos(w*f2*t)
		y= a2*np.sin(w*f2*t)
		z= 0.4-0.1*cos(w*f*t)
		pitch=-np.deg2rad(20)*np.sin(w*f*t)
		return (x,y,z,pitch)
	if name=='fig8':
		f,f2=(1/75,1/540)
		s=1.5
		a1,a2=(.8*s,1*s)
		offset=np.pi/4
		w=2*np.pi
		y=-a1*np.cos(w*f2*t+offset)*np.sin(w*f2*t+offset)
		x= -a2*np.cos(w*f2*t+offset)
		z= 0.35-0.15*np.cos(w*f*t)
		pitch=-np.deg2rad(35)*np.sin(w*f*t)
		return (x,y,z,pitch)
	if name=="test":
		return (0,0,0.4,np.deg2rad(-20))
	if name=="test2":
		w=np.pi*2
		f=1/120
		a=25
		return (0,0,0.35-0.2*np.cos(w*f*t),-np.deg2rad(a)*np.sin(w*f*t))
		
		
aprilKalP=0.5*np.eye(8)
aprilKalQ=np.diag([0.25 ,0.25 ,0.25,np.deg2rad(3),0.05,0.05,0.05,np.deg2rad(1.5)])
aprilKalRmeas1=0.02
aprilKalRmeas2=[0.1,0.1,0.1,np.deg2rad(10)**2]
aprilKalrdt=[0,1,2]
aprilKalcdt=[4,5,6]
cam_imu_tfvec=[0,0,-.2,-90,-90,0]
windowLen=15
windowTime=30#seconds

paramPath="calibrationData/modelParams.model"
PIDsavePath="Experiments/PID/"
BSsavePath="Experiments/Backstepping/"
ctrl_freq=10 #hz
ctrl_freqPID=10 #hz
cutoff=0.5
c=0*np.pi/9 #for backstepping controller
c2=np.pi/9#for xi record
#PIDgains=(.08,0,.1,1,0,0,.08,.001,.03)#pid gains->dkp,dki,dkd,tkp,tki,tkd,pkp,pki,pkd
#PIDgains=(.08,0,.1,1,0,0,.07,.003,.0375)#pid gains->dkp,dki,dkd,tkp,tki,tkd,pkp,pki,pkd
#PIDgains=(.08,0,.1,1,0,0,.5,.003,.0375)#pid gains->dkp,dki,dkd,tkp,tki,tkd,pkp,pki,pkd
PIDgains=(.08,0,.1,1,0.001,1,.1,.05,.0375)#pid gains->dkp,dki,dkd,tkp,tki,tkd,pkp,pki,pkd
#PIDgains=(1,0,10,1,0.001,1,1,0,2)#pid gains->dkp,dki,dkd,tkp,tki,tkd,pkp,pki,pkd
#backsteppingGains=(1,.05,2,1,1,1,1)#(1,1.25,2,1,.06,1,1) #backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)
#backsteppingGains=(1,.09,2,10,.9,.1,1)#(1,1.25,2,1,.06,1,1) #backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)
#backsteppingGains=(1,.5,1,1,5,2,2)#backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)
backsteppingGains=(1,.081,1,1,.99,.1,3)#backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)
backsteppingGains=(1,.08,4,1,.9,.1,4)#backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)
#backsteppingGains=(1,1,4,1,50,.1,4)#backstepping gains->k_o,k_z,k_xi,k_eta,k1(z),k2(eta),k3(xi)

trajList=['line','line2','pringle','ellipse','fig8','test','test2']
trajName=trajList[4]#trajList[1]
ExpLen=420

##camera setup##
#resx,resy=(320,240)
resx,resy=(640,480)
#resx,resy=(720,480)
#resx,resy=(800,608)
#resx,resy=(1088,720)

t_size=13.6/100 # m
pinhole = (608.14,609.30,322.16,234.34)#fx,fy,cx,cy
framerate = 60



Pd=0.5*np.eye(8)
Qd=np.diag([0.1 ,0.1 ,0.1,np.deg2rad(1),0.1,0.1,0.1,np.deg2rad(.5)])
Rmeasd=0*np.diag([0.001**2,0.001**2,0.001**2,np.deg2rad(.05)**2])
rdt=[0,1,2,3]
cdt=[4,5,6,7] 
