import math
import numpy as np
from numpy import sin, cos, tan, cross
import controllerHelper as ch

#actuator paramters
max_mpx=0.75 
min_mpx=0 
max_ppx=1 
min_ppx=0 
max_delta=np.deg2rad(90) 
min_delta=np.deg2rad(-90) 
ddeltaMin=-np.deg2rad(60)
ddeltaMax=np.deg2rad(60)

def kalmanUpdate(x,P,z,H,R):
	I=np.eye(P.shape[0])
	PHT=np.matmul(P,H.T) #PH'
	INV=np.linalg.inv(np.matmul(H,PHT)+R) #(HPH'+R)**-1
	K=np.matmul(PHT,INV) #PH'(HPH'+R)**-1
	x= x+np.matmul(K,z-np.matmul(H,x))#x+K(z-Hx)
	P= np.matmul(I-np.matmul(K,H),P)#(I-KH)P
	return x,P
    
def kalmanPrediction(x,u,A,B,P,Q):
	x=np.matmul(A,x) #Ax
	if type(B)==type(x):
		x=x+np.matmul(B,u) #Ax+Bu
	P=np.matmul(np.matmul(A,P),A.T)+Q #APA'+Q
	return x,P
	
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def eulerToRotm(alpha,beta,gamma):#input:roll,pitch,yaw. rotation sequuence: zyx 
    Rx = np.array([(1,0,0),(0,cos(alpha),sin(alpha)),(0,-sin(alpha),cos(alpha))])
    Ry = np.array([(cos(beta),0,-sin(beta)),(0,1,0),(sin(beta),0,cos(beta))])
    Rz = np.array([(cos(gamma),sin(gamma),0),(-sin(gamma),cos(gamma),0),(0,0,1)])
    matrix = np.matmul(np.matmul(Rz,Ry),Rx)
    return matrix	

def rot2eul(R):
	sy = np.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])
	singular = sy < 1e-6
	if  not singular:
		x = np.arctan2(R[2,1] , R[2,2])#*180.0/np.pi #roll
		y = np.arctan2(-R[2,0], sy)#*180.0/np.pi #pitch
		z = np.arctan2(R[1,0], R[0,0])#*180.0/np.pi #yaw
		return (x,y,z)
		
def Rbv(alpha,beta):
	return np.array([(cos(alpha)*cos(beta),-cos(alpha)*sin(beta),-sin(alpha)),(sin(beta),cos(beta),0),(sin(alpha)*cos(beta),-sin(alpha)*sin(beta),cos(beta))])	

def Sw(roll,pitch):
		return np.array([[1,tan(pitch)*sin(roll),tan(pitch)*cos(roll)],[0,cos(roll),-sin(roll)],[0,sin(roll)/cos(pitch),cos(roll)/cos(pitch)]])
		
def SMOCalc(z,zhat,Wb,Vihat,R,m0,delta,s,kz=50,ky=35,kx=35,eps=0.1,eps2=0.1,h1=2,h2=3,params = None):
    #sliding mode observers vor body fixed velocities
	#get model params
	if params==None:
	    M,g,rho,S,C_D0,C_alpha_D,C_delta_D,C_beta_FS,C_delta_FS,C_L0,C_alpha_L=(np.diag([8.0,19.8,10.8]),9.8,1000,0.019,0.45,17.59,1.43,-2,1.5,0.075,19.58)
	else:
	    M,g,rho,S,C_D0,C_alpha_D,C_delta_D,C_beta_FS,C_delta_FS,C_L0,C_alpha_L=params	
	k = np.asmatrix(np.array([(0),(0),(1)])).transpose()	

	#calculate intermediate variables
	Wb_skew = skew(Wb)
	V = np.sqrt(Vihat[0,0]**2+Vihat[1,0]**2+Vihat[2,0]**2)
	vz_hat=Vihat[2,0]
	vb_est=np.matmul(R.transpose(),Vihat)
	alpha=np.arctan2(vb_est[2,0],vb_est[0,0])
	if V == 0:
		beta = 0
	else:
		beta = np.arcsin(vb_est[1][0]/V)
	sat_e = s/eps
	if abs(sat_e)>1:
	    sat_e=np.sign(s)
	D = 0.5*rho*V*V*S*(C_D0 + C_alpha_D*alpha*alpha + C_delta_D*delta*delta)
	FS = 0.5*rho*V*V*S*(C_beta_FS*beta + C_delta_FS*delta)
	L = 0.5*rho*V*V*S*(C_L0 + C_alpha_L*alpha)
	F_ext_ = np.array([-D,FS,L])
	F_ext_.shape=(3,1)
	F_ext = np.matmul(Rbv(alpha,beta),F_ext_)
	innovation=np.array([kx*sat_e,ky*sat_e,kz*sat_e])
	innovation.shape=(3,1)
	#calculate derivatives
	temp=np.matmul(M,vb_est)
	temp = cross(temp[:,0], Wb[:,0])
	temp.shape=(3,1)
	dvb =  np.matmul(np.linalg.inv(M),temp + m0*g*np.matmul(R.transpose(),k) + F_ext)
	dVihat=np.matmul(np.matmul(R,Wb_skew),vb_est)+np.matmul(R,dvb)+innovation
	dzhat= ch.saturate(vz_hat+h1/eps2*(z-zhat),-2.0,2.0)
	dshat= h2/(eps2*eps2)*(z-zhat)+dzhat-dVihat[2,0]
	print(0*dshat,dzhat,z,zhat)
	return 0*dVihat,dzhat,0*dshat

def HGSMOCalc(z,zhat,Wb,Vihat,R,m0,delta,s,kz=50,ky=35,kx=35,eps=0.1,eps2=0.1,h1=2,h2=3,params = None):
    #high gain observers for inertial velocities
	#get model params
	if params==None:
	    M,g,rho,S,C_D0,C_alpha_D,C_delta_D,C_beta_FS,C_delta_FS,C_L0,C_alpha_L=(np.diag([8.0,19.8,10.8]),9.8,1000,0.019,0.45,17.59,1.43,-2,1.5,0.075,19.58)
	else:
	    M,g,rho,S,C_D0,C_alpha_D,C_delta_D,C_beta_FS,C_delta_FS,C_L0,C_alpha_L=params	
	k = np.asmatrix(np.array([(0),(0),(1)])).transpose()	

	#calculate intermediate variables
	Wb_skew = skew(Wb)
	V = np.sqrt(Vihat[0,0]**2+Vihat[1,0]**2+Vihat[2,0]**2)
	vz_hat=Vihat[2,0]
	vb_est=np.matmul(R.transpose(),Vihat)
	alpha=np.arctan2(vb_est[2,0],vb_est[0,0])
	if V == 0:
		beta = 0
	else:
		beta = np.arcsin(vb_est[1][0]/V)
	sat_e = s/eps
	if abs(sat_e)>1:
	    sat_e=np.sign(s)
	D = 0.5*rho*V*V*S*(C_D0 + C_alpha_D*alpha*alpha + C_delta_D*delta*delta)
	FS = 0.5*rho*V*V*S*(C_beta_FS*beta + C_delta_FS*delta)
	L = 0.5*rho*V*V*S*(C_L0 + C_alpha_L*alpha)
	F_ext_ = np.array([-D,FS,L])
	F_ext_.shape=(3,1)
	F_ext = np.matmul(Rbv(alpha,beta),F_ext_)
	innovation=np.array([kx*sat_e,ky*sat_e,kz*sat_e])
	innovation.shape=(3,1)
	#calculate derivatives
	temp=np.matmul(M,vb_est)
	temp = cross(temp[:,0], Wb[:,0])
	temp.shape=(3,1)
	dvb =  np.matmul(np.linalg.inv(M),temp + m0*g*np.matmul(R.transpose(),k) + F_ext)
	dVihat=np.matmul(np.matmul(R,Wb_skew),vb_est)+np.matmul(R,dvb)+innovation
	dzhat= ch.saturate(vz_hat+h1/eps2*(z-zhat),-2.0,2.0)
	dshat= h2/(eps2*eps2)*(z-zhat)+dzhat-dVihat[2,0]
	return 0*dshat,dzhat,z,zhat
	
def velEstimator(X,R,u,vb_est,zhat,params,K=[1,1,1]): 
	#body-fixed velocity observer usig only depth
	#only considers gliding motion
	z = X[2] 
	w1,w2,w3 = X[6:9]
	
	# r11 = X(10)  r12 = X(11)  r13 = X(12)  
	# r21 = X(13)  r22 = X(14)  r23 = X(15)  
	# r31 = X(16)  r32 = X(17)  r33 = X(18) 

	body_accel=np.array([[X[3]],[X[4]],[X[5]]]) 
	ppx = u[1] 
	delta = u[2]
	

	# unpack needed model parameters
	mc,lm,bc,lp,g,m1,m2,m3,_,CD0,CaD,CdD,C_beta_FS,C_delta_FS,CL0,CaL,_,_,_,_,_,_,_,_,_,_,_,S,_,_,rho=params
	# m1 = params(6)  m2 = params(7)  m3 = params(8)  
	# CD0 = params(10)   CaD = params(11)  CdD = params(12)  
	# C_beta_FS = params(13)  C_delta_FS = params(14)  
	# CL0 = params[15]  CaL = params(16)  rho = params(31)  
	# S = params[28]  g=params[5] 
	# mc=params[1]  
	# lm=params[2] #meters
	# bc=params[3]  
	# lp=params[4]  #meters*g/mm

	# calculate m0 
	m0= lp*(ppx-bc) 

	# R = [r11, r12, r13  r21, r22, r23  r31, r32, r33] 
	omega_b = np.array([[w1],[w2], [w3]]) 
	#     R
	#     rotm2eul(R)
	#     atan2(r21,r11)

	# caclculate body fixed velocities
	v1 = vb_est[0,0] 
	v2 = vb_est[1,0] 
	v3 = vb_est[2,0] 
	#calculate intermediate variables
	V = np.sqrt(v1**2 + v2**2 + v3**2) 
	
	alpha = np.arctan2(v3,v1) 
	beta=0 if v2==0 else np.arcsin(v2/V)
	

	R_bv11 = np.cos(alpha)*np.cos(beta) 
	R_bv12 = -np.cos(alpha)*np.sin(beta) 
	R_bv13 = -np.sin(alpha) 
	R_bv21 = np.sin(beta) 
	R_bv22 = np.cos(beta) 
	R_bv23 = 0 
	R_bv31 = np.sin(alpha)*np.cos(beta) 
	R_bv32 = -np.sin(alpha)*np.sin(beta) 
	R_bv33 = np.cos(alpha) 

	D = 0.5*rho*V**2*S*(CD0 + CaD*alpha**2 + CdD*delta**2)  
	FS = 0.5*rho*V**2*S*(C_beta_FS*beta + C_delta_FS*delta) 
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)
	L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*np.cos(alpha)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))*np.cos(alpha) 

	R_bv = np.array([[R_bv11,R_bv12,R_bv13],[R_bv21,R_bv22,R_bv23],[R_bv31,R_bv32,R_bv33]]) 
	F_ext = np.matmul(R_bv,np.array([[-D],[FS],[-L]]))
	M = np.diag([m1,m2,m3]) 
	k = np.array([[0],[0],[1]]) 
	v_b_dot =np.matmul(np.linalg.inv(M), np.cross(np.matmul(M,vb_est), omega_b,axis=0) + m0*g*np.matmul(R.transpose(),k) + F_ext)  # 3x1
	
	K=np.diag(K)
	dPos_est = np.matmul(R,vb_est)+.5*np.array([[0],[0],[z-zhat]])
	dvb_est = v_b_dot+np.matmul(K,np.matmul(R.transpose(),np.array([[0],[0],[z-zhat]])))
	#dPos_est = R*vb_est+0*diag(K)*[0,0,z-zhat]
	#dvb_est = v_b_dot+diag(K)*(-(body_accel-v_b_dot)+R.transpose()*[0,0,z-zhat]
	return dPos_est,dvb_est

    

def velEstimator2(X,R,u,vb_est,zhat,params,K=[1,1,1]):
	#body-fixed velocity observer full position 
	#only considers gliding motion
	x_pos = X[0]  
	y_pos = X[1]  
	z = X[2] 
	w1,w2,w3 = X[6:9]
	# r11 = X(10)  r12 = X(11)  r13 = X(12)  
	# r21 = X(13)  r22 = X(14)  r23 = X(15)  
	# r31 = X(16)  r32 = X(17)  r33 = X(18) 

	body_accel=np.array([[X[3]],[X[4]],[X[5]]]) 
	ppx = u[1] 
	delta = u[2]

	# unpack need model parameters
	mc,lm,bc,lp,g,m1,m2,m3,_,CD0,CaD,CdD,C_beta_FS,C_delta_FS,CL0,CaL,_,_,_,_,_,_,_,_,_,_,_,S,_,_,rho=params
	# m1 = params(6)  m2 = params(7)  m3 = params(8)  
	# CD0 = params(10)   CaD = params(11)  CdD = params(12)  
	# C_beta_FS = params(13)  C_delta_FS = params(14)  
	# CL0 = params[15]  CaL = params(16)  rho = params(31)  
	# S = params[28]  g=params[5] 
	# mc=params[1]  
	# lm=params[2] #meters
	# bc=params[3]  
	# lp=params[4]  #meters*g/mm

	# calculate m0 
	m0= lp*(ppx-bc) 

	# R = [r11, r12, r13  r21, r22, r23  r31, r32, r33] 
	omega_b = np.array([[w1],[w2], [w3]])     
	#     R
	#     rotm2eul(R)
	#     atan2(r21,r11)

	# caclculate body fixed velocities
	v1 = vb_est[0,0] 
	v2 = vb_est[1,0] 
	v3 = vb_est[2,0] 
	#calculate intermediate variables
	V = np.sqrt(v1**2 + v2**2 + v3**2) 
	alpha = np.arctan2(v3,v1) 
	beta=0 if V==0 else np.arcsin(v2/V)
	R_bv11 = np.cos(alpha)*np.cos(beta) 
	R_bv12 = -np.cos(alpha)*np.sin(beta) 
	R_bv13 = -np.sin(alpha) 
	R_bv21 = np.sin(beta) 
	R_bv22 = np.cos(beta) 
	R_bv23 = 0 
	R_bv31 = np.sin(alpha)*np.cos(beta) 
	R_bv32 = -np.sin(alpha)*np.sin(beta) 
	R_bv33 = np.cos(alpha) 

	D = 0.5*rho*V**2*S*(CD0 + CaD*alpha**2 + CdD*delta**2)  
	FS = 0.5*rho*V**2*S*(C_beta_FS*beta + C_delta_FS*delta) 
	#L = 0.5*rho*V**2*S*(CL0 + CaL*alpha)
	L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*np.cos(alpha)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))*np.cos(alpha) 

	R_bv = np.array([[R_bv11,R_bv12,R_bv13],[R_bv21,R_bv22,R_bv23],[R_bv31,R_bv32,R_bv33]]) 
	F_ext = np.matmul(R_bv,np.array([[-D],[FS],[-L]]))
	M = np.diag([m1,m2,m3]) 
	k = np.array([[0],[0],[1]]) 
	v_b_dot =np.matmul(np.linalg.inv(M), np.cross(np.matmul(M,vb_est), omega_b,axis=0) + m0*g*np.matmul(R.transpose(),k) + F_ext)  # 3x1

	K=np.diag(K)
	#dPos_est = np.matmul(R,vb_est)+np.matmul(0.5*K,np.matmul(R.transpose(),np.array([[x_pos],[y_pos],[z]])-pos))
	dPos_est = np.matmul(R,vb_est)+.5*np.array([[0],[0],[z-pos[2,0]]])#+np.matmul(0.5*K,np.array([[x_pos],[y_pos],[z]])-pos)
	dvb_est = v_b_dot+np.matmul(K,np.matmul(R.transpose(),np.array([[x_pos],[y_pos],[z]])-pos))
	#dPos_est = R*vb_est+0*diag(K)*[[x_pos y_pos z]-pos] 
	#dvb_est = v_b_dot+diag(K)*(-(body_accel-v_b_dot)+R'*[[x_pos y_pos z]-pos]) 
	return dPos_est,dvb_est

def vytEstimator(X,R,u,vb_est,zhat,delta_hat,params,K=[1,1,1]): 
	#observer for body-fixed velocity, yaw angle, and tail angle
	z = X[2] 
	w1,w2,w3 = X[6:9]
	
	# r11 = X(10)  r12 = X(11)  r13 = X(12)  
	# r21 = X(13)  r22 = X(14)  r23 = X(15)  
	# r31 = X(16)  r32 = X(17)  r33 = X(18) 

	#mpx=u[0]
	ppx = u[1] 
	deltad = u[2]
	

	# unpack need model parameters
	mc,lm,bc,lp,g,m1,m2,m3,mbar,CD0,CaD,CdD,C_beta_FS,C_delta_FS,CL0,CaL,J1,J2,J3,CM0,C_beta_MR,CaMrho,C_beta_MY,C_delta_MY,Kq1,Kq2,Kq3,S,mw,rw3,rho,k_delta,ddeltamm,lc,ld,S_delta,dJ1,dJ2,dJ3=params
	ddeltaMin,ddeltaMax=(-ddeltamm,ddeltamm)
	# m1 = params(6) #kg; m2 = params(7) #kg; m3 = params(8) #kg; mbar = params(9); #kg
	#CD0 = params(10); CaD = params(11) #rad^-2; CdD = params(12) #rad^-2; 
	#C_beta_FS = params(13) #rad^-1; C_delta_FS = params(14) #rad^-1; CL0 = params(15);
	#CaL = params(16) #rad^-1; J1 = params(17) #kg*m^2; J2 = params(18) #kg*m^2; J3 = params(19) #kg*m^2;
	#CM0 = params(20) #m; 	C_beta_MR = params(21) #m/rad; CaMrho = params(22) #m/rad;
	#C_beta_MY = params(23) #m/rad; C_delta_MY = params(24) #m/rad;
	#Kq1 = params(25) #m*s/rad; Kq2 = params(26) #m*s/rad; Kq3 = params(27) #m*s/rad; S = params(28) #m^2;
	#mw = params(29) #kg; rw3 = params(30) #m; rho = params(31) #kg/m^3;
	#k_delta=params(32) #rad/s; ddelta_min=-params(33)#tail angle min velocity; ddelta_max= params(33)# tail angle max velocity;
	#lc=params(34);ld=params(35)# length between tail fin and center of mass;
	#S_delta=params(36)#surface area of tail fin; 
	#dJ1=params(37); dJ2=params(38); dJ3=params(39);
	

	# calculate m0 
	#rp1=lm*(mpx-mc)
	m0= lp*(ppx-bc) 
	delta_dot=ch.saturate(k_delta*(delta_d-delta),ddelta_min,ddelta_max) #k_delta*(delta_d-delta);#

	# R = [r11, r12, r13  r21, r22, r23  r31, r32, r33] 
	omega_b = np.array([[w1],[w2], [w3]]) 
	#     R
	#     rotm2eul(R)
	#     atan2(r21,r11)

	# caclculate body fixed velocities
	v1 = vb_est[0,0] 
	v2 = vb_est[1,0] 
	v3 = vb_est[2,0] 
	#calculate intermediate variables
	V = np.sqrt(v1**2 + v2**2 + v3**2) 
	
	alpha = np.arctan2(v3,v1) 
	beta=0 if v2==0 else np.arcsin(v2/V)
	

	R_bv11 = np.cos(alpha)*np.cos(beta) 
	R_bv12 = -np.cos(alpha)*np.sin(beta) 
	R_bv13 = -np.sin(alpha) 
	R_bv21 = np.sin(beta) 
	R_bv22 = np.cos(beta) 
	R_bv23 = 0 
	R_bv31 = np.sin(alpha)*np.cos(beta) 
	R_bv32 = -np.sin(alpha)*np.sin(beta) 
	R_bv33 = np.cos(alpha) 

	D = 0.5*rho*V**2*S*(CD0 + CaD*alpha**2 + CdD*delta**2)  
	FS = 0.5*rho*V**2*S*(C_beta_FS*beta + C_delta_FS*delta) 
	#L = 0.5*rho*V**2*S*(CL0 + CaL*alpha)
	L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*np.cos(alpha)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))*np.cos(alpha) 
	Ftail = lc*rho*S_delta*-k_delta*delta_dot*np.array([[sin(delta)],[cos(delta)],[0]])
	

	R_bv = np.array([[R_bv11,R_bv12,R_bv13],[R_bv21,R_bv22,R_bv23],[R_bv31,R_bv32,R_bv33]]) 
	F_ext = np.matmul(R_bv,np.array([[-D],[FS],[-L]]))+Ftail
	M = np.diag([m1,m2,m3]) 
	k = np.array([[0],[0],[1]]) 
	v_b_dot =np.matmul(np.linalg.inv(M), np.cross(np.matmul(M,vb_est), omega_b,axis=0) + m0*g*np.matmul(R.T,k) + F_ext)  # 3x1
	
	K=np.diag(K)
	dPos_est = np.matmul(R,vb_est)+.5*np.array([[0],[0],[z-zhat]])
	dvb_est = v_b_dot+np.matmul(K,np.matmul(R.T,np.array([[0],[0],[z-zhat]])))
	ddelta=ch.saturate(kdelta*(deltad-deltahat),ddeltaMin,ddeltaMax)
	return dPos_est,dvb_est,ddelta,dyaw
	
def vytwEstimator(meas,R,u,vb_est,wb_est,zhat,delta_hat,params,K=[1,1,1],Kw=[1,1,1]): 
	#observer for body-fixed velocity, yaw angle, and tail angle, angular velocities
	z,w1,w2,w3 = meas
	w1h,w2h,w3h = wb_est[:,0].tolist()
	
	# r11 = X(10)  r12 = X(11)  r13 = X(12)  
	# r21 = X(13)  r22 = X(14)  r23 = X(15)  
	# r31 = X(16)  r32 = X(17)  r33 = X(18) 

	mpx=u[0]
	ppx = u[1] 
	deltad = u[2]
	

	# unpack need model parameters
	mc,lm,bc,lp,g,m1,m2,m3,mbar,CD0,CaD,CdD,C_beta_FS,C_delta_FS,CL0,CaL,J1,J2,J3,CM0,C_beta_MR,CaMrho,C_beta_MY,C_delta_MY,Kq1,Kq2,Kq3,S,mw,rw3,rho,k_delta,ddeltamm,lc,ld,S_delta,dJ1,dJ2,dJ3,ld2=params
	ddeltaMin,ddeltaMax=(-ddeltamm,ddeltamm)
	# m1 = params(6) #kg; m2 = params(7) #kg; m3 = params(8) #kg; mbar = params(9); #kg
	#CD0 = params(10); CaD = params(11) #rad^-2; CdD = params(12) #rad^-2; 
	#C_beta_FS = params(13) #rad^-1; C_delta_FS = params(14) #rad^-1; CL0 = params(15);
	#CaL = params(16) #rad^-1; J1 = params(17) #kg*m^2; J2 = params(18) #kg*m^2; J3 = params(19) #kg*m^2;
	#CM0 = params(20) #m; 	C_beta_MR = params(21) #m/rad; CaMrho = params(22) #m/rad;
	#C_beta_MY = params(23) #m/rad; C_delta_MY = params(24) #m/rad;
	#Kq1 = params(25) #m*s/rad; Kq2 = params(26) #m*s/rad; Kq3 = params(27) #m*s/rad; S = params(28) #m^2;
	#mw = params(29) #kg; rw3 = params(30) #m; rho = params(31) #kg/m^3;
	#k_delta=params(32) #rad/s; ddelta_min=-params(33)#tail angle min velocity; ddelta_max= params(33)# tail angle max velocity;
	#lc=params(34);ld=params(35)# length between tail fin and center of mass;
	#S_delta=params(36)#surface area of tail fin; 
	#dJ1=params(37); dJ2=params(38); dJ3=params(39);
	

	# calculate m0 
	rp1=lm*(mpx-mc)
	m0= lp*(ppx-bc)
	delta_dot=ch.saturate(k_delta*(deltad-delta_hat),ddeltaMin,ddeltaMax) 


	# R = [r11, r12, r13  r21, r22, r23  r31, r32, r33] 
	omega_b = np.array([[w1],[w2], [w3]]) 
	omega_bt = np.array([[w1h],[w2h], [w3h]]) 
	#     R
	#     rotm2eul(R)
	#     atan2(r21,r11)

	# body fixed velocities
	v1 = vb_est[0,0] 
	v2 = vb_est[1,0] 
	v3 = vb_est[2,0] 
	#calculate intermediate variables
	V = np.sqrt(v1**2 + v2**2 + v3**2) 
	
	alpha = np.arctan2(v3,v1) 
	beta=0 if V==0 else np.arcsin(v2/V)
	

	R_bv11 = np.cos(alpha)*np.cos(beta) 
	R_bv12 = -np.cos(alpha)*np.sin(beta) 
	R_bv13 = -np.sin(alpha) 
	R_bv21 = np.sin(beta) 
	R_bv22 = np.cos(beta) 
	R_bv23 = 0 
	R_bv31 = np.sin(alpha)*np.cos(beta) 
	R_bv32 = -np.sin(alpha)*np.sin(beta) 
	R_bv33 = np.cos(alpha) 
	
	delta=delta_hat
	D = 0.5*rho*V**2*S*(CD0 + CaD*alpha**2 + CdD*delta**2)  
	FS = 0.5*rho*V**2*S*(C_beta_FS*beta + C_delta_FS*delta)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*cos(alpha)
	L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*np.cos(alpha)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))*np.cos(alpha)
	Ftail = lc*rho*S_delta*-k_delta*delta_dot*np.array([[sin(delta)],[ld2*cos(delta)],[0]])
	M1 = 0.5*rho*V**2*S*(C_beta_MR*beta + Kq1*w1)
	M2 = 0.5*rho*V**2*S*(CM0 + CaMrho*alpha + Kq2*w2)
	M3 = 0.5*rho*V**2*S*(C_beta_MY*beta + Kq3*w3 + C_delta_MY*delta)
	Mtail = ld*Ftail*np.array([[0],[0],[cos(delta)]])


	R_bv = np.array([[R_bv11,R_bv12,R_bv13],[R_bv21,R_bv22,R_bv23],[R_bv31,R_bv32,R_bv33]]) 
	F_ext = np.matmul(R_bv,np.array([[-D],[FS],[-L]]))+Ftail
	T_ext = np.matmul(R_bv,np.array([[M1],[M2],[M3]]))+Mtail
	M = np.diag([m1,m2,m3]) 
	J = np.diag([J1,J2,J3])
	dJ= np.diag([dJ1,dJ2,dJ3])
	k = np.array([[0],[0],[1]]) 
	r_w = np.array([[0],[0],[rw3]])
	rp = np.array([[rp1],[0],[0]])
	K=np.diag(K)
	v_b_dot =np.matmul(np.linalg.inv(M), np.cross(np.matmul(M,vb_est), omega_b,axis=0) + m0*g*np.matmul(R.T,k) + F_ext)  # 3x1
	w_b_dot=np.matmul(np.linalg.inv(J), np.matmul(-dJ,omega_bt) + np.cross(np.matmul(J,omega_bt), omega_bt,axis=0) + np.cross(np.matmul(M,vb_est), vb_est,axis=0)+ T_ext  + np.cross(mw*g*r_w,np.matmul(R.T,k),axis=0) + np.cross(mbar*g*rp,np.matmul(R.T,k),axis=0))  # 3x1
	roll,pitch,yaw=rot2eul(R)
	
	dPos_est = np.matmul(R,vb_est)+.5*np.array([[0],[0],[z-zhat]])
	dvb_est = v_b_dot+np.matmul(K,np.matmul(R.T,np.array([[0],[0],[z-zhat]])))
	psi_dot = np.matmul(Sw(roll,pitch),omega_b)
	dwb_est=w_b_dot+1*np.matmul(np.diag(Kw),(omega_b-omega_bt))
	ddelta=ch.saturate(k_delta*(deltad-delta_hat),ddeltaMin,ddeltaMax)
	
	return dPos_est,dvb_est,ddelta,psi_dot,dwb_est

def fullStateObserver(est,meas,u,params,K,asList=True):
	## parse states
	x_pos = x[0] 
	y_pos = x[1] 
	z_pos = x[2] 
	yaw = x[3]
	pitch = x[4] 
	roll = x[5]
	v1 = x[6] 
	v2 = x[7] 
	v3 = x[8] 
	w1 = x[9] 
	w2 = x[10] 
	w3 = x[11]  
	#delta=max(min(max_delta,x[12]),min_delta) 
	#mpx=max(min(max_ppx,x[13]),min_ppx) 
	#ppx=max(min(max_mpx,x[14]),min_mpx) 
	delta=ch.saturate(x[12],min_delta,max_delta) 
	mpx=ch.saturate(x[13],min_ppx,max_ppx) 
	ppx=ch.saturate(x[14],min_mpx,max_mpx) 
	V =np.sqrt(v1**2 + v2**2 + v3**2) 
	alpha = atan2(v3,v1) 
	if V==0:
		beta=0 
	else:
		beta = asin(v2/V) 
	
	R = eulerToRotm(roll,pitch,yaw)
	v_b = np.array([[v1],[v2],[v3]])
	omega_b = np.array([[w1],[w2],[w3]])
	omega_b_hat = skew([w1, w2, w3])

	## parse inputs
	mpx_dot = u[0]
	ppx_dot = u[1]
	b       = u[2]#tail bias
	amp     = u[3]#tail amplitude
	freq    = u[4]#tail frequency
	delta_d = b+amp*sin(2*np.pi*freq*t)

	# unpack model parameters
	mc,lm,bc,lp,g,m1,m2,m3,mbar,CD0,CaD,CdD,C_beta_FS,C_delta_FS,CL0,CaL,J1,J2,J3,CM0,C_beta_MR,CaMrho,C_beta_MY,C_delta_MY,Kq1,Kq2,Kq3,S,mw,rw3,rho,k_delta,ddeltamm,lc,ld,S_delta,dJ1,dJ2,dJ3,ld2=params
	ddelta_min,ddelta_max=(-ddeltamm,ddeltamm)
	# m1 = params(6) #kg; m2 = params(7) #kg; m3 = params(8) #kg; mbar = params(9); #kg
	#CD0 = params(10); CaD = params(11) #rad^-2; CdD = params(12) #rad^-2; 
	#C_beta_FS = params(13) #rad^-1; C_delta_FS = params(14) #rad^-1; CL0 = params(15);
	#CaL = params(16) #rad^-1; J1 = params(17) #kg*m^2; J2 = params(18) #kg*m^2; J3 = params(19) #kg*m^2;
	#CM0 = params(20) #m; 	C_beta_MR = params(21) #m/rad; CaMrho = params(22) #m/rad;
	#C_beta_MY = params(23) #m/rad; C_delta_MY = params(24) #m/rad;
	#Kq1 = params(25) #m*s/rad; Kq2 = params(26) #m*s/rad; Kq3 = params(27) #m*s/rad; S = params(28) #m^2;
	#mw = params(29) #kg; rw3 = params(30) #m; rho = params(31) #kg/m^3;
	#k_delta=params(32) #rad/s; ddelta_min=-params(33)#tail angle min velocity; ddelta_max= params(33)# tail angle max velocity;
	#lc=params(34);ld=params(35)# length between tail fin and center of mass;
	#S_delta=params(36)#surface area of tail fin; 
	#dJ1=params(37); dJ2=params(38); dJ3=params(39);
	
	#
	rp1=lm*(mpx-mc)
	m0= lp*(ppx-bc)
	#delta_dot=min(max(ddelta_min,k_delta*(delta_d-delta)),ddelta_max) #k_delta*(delta_d-delta);#
	delta_dot=ch.saturate(k_delta*(delta_d-delta),ddelta_min,ddelta_max) #k_delta*(delta_d-delta);#

	rp = np.array([[rp1],[0],[0]])

	M = np.diag([m1,m2,m3]) #M = (ms + mbar)*I + M_f
	J = np.diag([J1,J2,J3])
	dJ= np.diag([dJ1,dJ2,dJ3])
	r_w = np.array([[0],[0],[rw3]])
	k = np.array([[0],[0],[1]]) 
	
	
	## Define external forces and torques
	D  = 0.5*rho*V**2*S*(CD0 + CaD*alpha**2 + CdD*delta**2)
	FS = 0.5*rho*V**2*S*(C_beta_FS*beta + C_delta_FS*delta)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)
	L  = 0.5*rho*V**2*S*(CL0 + CaL*alpha)*np.cos(alpha)
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))
	#L  = 0.5*rho*V**2*S*(CL0 + CaL*np.sin(alpha))*np.cos(alpha)
	Ftail = lc*rho*S_delta*-k_delta*delta_dot*np.array([[sin(delta)],[ld2*cos(delta)],[0]])
	M1 = 0.5*rho*V**2*S*(C_beta_MR*beta + Kq1*w1)
	M2 = 0.5*rho*V**2*S*(CM0 + CaMrho*alpha + Kq2*w2)
	M3 = 0.5*rho*V**2*S*(C_beta_MY*beta + Kq3*w3 + C_delta_MY*delta)
	Mtail = ld*lc*rho*S_delta*-k_delta*delta_dot*np.array([[0],[0],[cos(delta)]])
	
	R_bv = Rbv(alpha,beta)
	F_ext = np.matmul(R_bv,np.array([[-D],[FS],[-L]]))+Ftail
	T_ext = np.matmul(R_bv,np.array([[M1],[M2],[M3]]))+Mtail

	## Define state dynamics
	b_i_dot = np.matmul(R,v_b) # 3x1
	psi_dot = np.matmul(Sw(roll,pitch),omega_b)
	v_b_dot =np.matmul(np.linalg.inv(M), np.cross(np.matmul(M,v_b), omega_b,axis=0) + m0*g*np.matmul(R.T,k) + F_ext)+np.matmul(K,np.matmul(R.T,np.array([[0],[0],[z-zhat]])))  # 3x1
	omega_b_dot =np.matmul(np.linalg.inv(J), np.matmul(-dJ,omega_b) + np.cross(np.matmul(J,omega_b), omega_b,axis=0) + np.cross(np.matmul(M,v_b), v_b,axis=0)+ T_ext  + np.cross(mw*g*r_w,np.matmul(R.T,k),axis=0) + np.cross(mbar*g*rp,np.matmul(R.T,k),axis=0))  # 3x1
	act_Vec=np.array([[delta_dot],[mpx_dot],[ppx_dot]])
	dx = np.concatenate((b_i_dot,psi_dot,v_b_dot,omega_b_dot,act_Vec))
	if asList:
		return dx[:,0].tolist()
	return dx