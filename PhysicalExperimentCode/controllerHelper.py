import numpy as np
import threading 
import time 
import socket

FRESH_WATER = 0
SALT_WATER = 1

def requestControl(sock):
	sock.send("requestControl,".encode("utf-8"))
	return sock.recv(1024)
	
def readDepthI2C(sock,mod=FRESH_WATER):
	if mod == None:
		sock.send("R,depth,\n".encode("utf-8"))
	elif mod == FRESH_WATER:
		sock.send("R,depthFresh,\n".encode("utf-8"))
	elif mod == SALT_WATER:
		sock.send("R,depthOcean,\n".encode("utf-8"))
	ans=sock.recv(1024).decode("utf-8")
	if ans == b"not available":
		return -10	
	return float(ans)
	
def readDepth(sock):
	sock.send("R,depth,\n".encode("utf-8"))
	ans=sock.recv(1024).decode("utf-8")
	if ans == b"not available":
		return -10
	return float(ans)

def readYaw(sock):
	sock.send("R,yaw,\n".encode("utf-8"))
	return float(sock.recv(1024).decode("utf-8"))	
	
def readPitch(sock):
	sock.send("R,pitch,\n".encode("utf-8"))
	return float(sock.recv(1024).decode("utf-8"))
	
def readRoll(sock):
	sock.send("R,roll,\n".encode("utf-8"))
	return float(sock.recv(1024).decode("utf-8"))

def readEuler(sock,units="rad"):
	if units =="rad":
		sock.send("R,rpy_rad,\n".encode("utf-8"))
	else:
		sock.send("R,rpy,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	
	
	
def readMagRaw(sock): #raw values
	sock.send("R,mag,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])		
	
def readAccelRaw(sock): #raw values
	sock.send("R,accel,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	
	
def readGyroRaw(sock): #raw values
	sock.send("R,gyro,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	

def readMag(sock): #compensated values
	sock.send("R,magComp,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])		
	
def readAccel(sock): #compensated values
	sock.send("R,accelComp,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	
	
def readGyro(sock): #compensted values
	sock.send("R,gyroComp_rad,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	

def readBodyAccel(sock): #body fixed acceleration estimate
	sock.send("R,b_accel,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2])	
	
def readIMU(sock):
	sock.send("R,imuComp,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),float(data[8])	
def readIMURaw(sock):
	sock.send("R,imu,\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8").split(',')
	return float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),float(data[8])	
	
def readBattVolt(sock):
	sock.send("R,battVolt\n".encode("utf-8"))
	data=sock.recv(1024).decode("utf-8")
	return float(data)	
	
def setAllActsPos(sock,angle=-360,massPos=-1,pumpPos=-1):#mass%,pump%,servo angle	
	if angle==-360 and massPos==-1 and PumpPos==-1:
		return
	sock.send(str.encode("S,inputsPos,"+str(massPos)+","+str(pumpPos)+","+str(int(round(angle)))+",\n","utf-8"))

def setAllActsSpd(sock,angle=-360,massSpd=-1,PumpSpd=-1):#mass%,pump%,servo angle	
	if angle==-360 and massSpd==-1 and massSpd==-1:
		return
	sock.send(str.encode("S,inputsSpd,"+str(massSpd)+","+str(PumpSpd)+","+str(int(round(angle)))+",\n","utf-8"))
	
def backsteppingCmd(sock,rp1,m0,par,delta=-360):
	offset,scale=(par[0],par[1])
	mp =saturate(rp1/scale+offset,0,.95)*100
	offset2,scale2=(par[2],par[3])
	pp =saturate(m0/scale2+offset2,0,1)*100
	setAllActsPos(sock,massPos=mp,pumpPos=pp,angle=delta)
	
def rp1ToActPos(rp1,par):
	offset,scale=(par[0],par[1])
	mp =saturate(rp1/scale+offset,0,.95)*100
	return mp
	
def m0ToActPos(m0,par):
	offset2,scale2=(par[2],par[3])
	pp =saturate(m0/scale2+offset2,0,1)*100
	return pp
		
def readRGB(RGBSock):
	RGBSock.send("R,rgb,\n".encode("utf-8"))
	d=RGBSock.recv(1024).decode("utf-8")
	t,r,g,b=d.split(",")
	return float(t),float(r),float(g),float(b)


def shutRGB(RGBSock):
	RGBSock.send("shutdown,\n".encode("utf-8"))
		
def setServoAngle(sock,angle):
	sock.send(str.encode("S,servo,"+str(int(round(angle)))+",\n","utf-8"))
	
def setMassPos(sock,per):
	sock.send(str.encode("S,mass%,"+str(per)+",\n","utf-8"))

def setRp1(sock,rp1,par):
	offset,scale=(par[0],par[1])
	per =saturate(rp1/scale+offset,0,.95)   
	setMassPos(sock,per*100)
		
def setMassSpd(sock,spd):
	sock.send(str.encode("S,massSpd,"+str(spd)+",\n","utf-8"))	

def setPumpPos(sock,per):
	sock.send(str.encode("S,pump%,"+str(per)+",\n","utf-8"))

def setM0(sock,m0,par):
	offset,scale=(par[2],par[3])
	per =saturate(m0/scale+offset,0,1) 
	setPumpPos(sock,per*100)
		
def setPumpSpd(sock,spd):
	sock.send(str.encode("S,pumpSpd,"+str(spd)+",\n","utf-8"))	

def readPumpPos(sock):
	sock.send(str.encode("R,pump%,\n","utf-8"))
	return float(sock.recv(1024).decode("utf-8"))
	
def readM0(sock):
	offset,scale=(params.pumpLen,params.COB)
	return (readPumpPos(sock) - offset)*scale 
	
def readMassPos(sock):	
	sock.send(str.encode("R,mass%,\n","utf-8"))
	return float(sock.recv(1024).decode("utf-8"))
	
def readInputs(sock):#mass%,pump%,servo angle	
	sock.send(str.encode("R,inputs,\n","utf-8"))
	vals=sock.recv(1024).decode("utf-8").split(',')
	return float(vals[0]),float(vals[1]),float(vals[2])
	
def readRp1(sock):
	scale,offset=(params.massLen,params.COM)
	return (readMassPos(sock) - offset)*scale 
		
def readServoPos(sock):	
	sock.send(str.encode("R,servo,\n","utf-8"))
	return float(sock.recv(1024).decode("utf-8"))
	
			
def yawCorrection(yaw, yaw_d,wrapVal,minVal=-70,maxVal=70,k=1):
	#correction1 = ((yaw+wrapVal)%wrapVal)-((yaw_d+wrapVal)%wrapVal)
	#correction = yaw-yaw_d
	#if (abs(correction)>abs(correction1)):
	#	correction = correction1
	correction=angleWrap(yaw-yaw_d,wrapVal)
	return min(max(k*correction,minVal),maxVal)

def simpleLPF(x,lastFilterState,r):
	newFilterState = r*x+(1-r)*lastFilterState
	return newFilterState
	
def saturate(x,lower,upper):
	return max(min(x,upper),lower)
	
def angleWrap(angle,wrapVal):
	return (angle+wrapVal)%(2.0*wrapVal)-wrapVal 

def eul2rotm(alpha,beta,gamma):#input:x,y,z. rotation sequuence: zyx 
    Rx = np.array([(1,0,0),(0,cos(alpha),sin(alpha)),(0,-sin(alpha),cos(alpha))])
    Ry = np.array([(cos(beta),0,-sin(beta)),(0,1,0),(sin(beta),0,cos(beta))])
    Rz = np.array([(cos(gamma),sin(gamma),0),(-sin(gamma),cos(gamma),0),(0,0,1)])
    matrix = np.matmul(np.matmul(Rz,Ry),Rx)
    return matrix
    
def GPS_GetBearingDistanceToTarget(lattitude, longitude, targetLatitude, targetLongitude):
	R =6371000.0 #//radius of the Earth
	lat1 = np.deg2rad(lattitude)
	lat2 = np.deg2rad(targetLatitude)
	lon1 = np.deg2rad(longitude) #// negative sign for Western Hemisphere
	lon2 = np.deg2rad(targetLongitude) #// negative sign for Western Hemisphere
	dLat = lat2 - lat1
	dLon = lon2 - lon1
	# reference: http://www.movable-type.co.uk/scripts/latlong.html
	# compute target bearing (angle from north)
	y = sin(dLon)*cos(lat2)
	x = cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dLat)
	dGPSBearing = np.rad2deg(atan2(y,x))
	# reference: http://www.movable-type.co.uk/scripts/latlong.html
	# compute distance from target				
	a = sin(dLat/2) * sin(dLat/2) + cos(lat1)*cos(lat2)*sin(dLon/2)*sin(dLon/2)
	c = 2*atan2(sqrt(a),sqrt(1-a))
	dGPSDist = R*c
	return dGPSBearing,dGPSDist

def converTGPSFormat(lat,lon):#change LAT,LONG format from ddmm.mmmm to dd.mmmmmm
	return lat/100.0,lon/100.0
	
class PID():
	def __init__(self,kp=1,ki=1,kd=1,clip=None,smoothingFactor=0.8):
		self.kp=kp
		self.ki=ki
		self.kd=kd
		self.r=max(smoothingFactor,1e-4)
		self.lpfTerm=0
		self.highClip=None
		self.lowClip=None
		self.saturateIntegral = False
		if clip !=None:
			self.saturateIntegral =True
			self.highClip=clip[1]
			self.lowClip=clip[0]
		self.lastErr=0
		self.sumErr=0
		
	def run(self,e,dt):
		self.sumErr += e*dt
		if self.saturateIntegral and self.lowClip != None and self.highClip != None:
			self.sumErr= min(max(self.sumErr,self.lowClip),self.highClip)
		if self.r<1:
			derTerm=((self.r)*(e-self.lastErr)/dt+(1-self.r)*self.lpfTerm)
			self.lpfTerm=derTerm
		else:
			derTerm=(e-self.lastErr)/dt
		u = self.kp*e+self.ki*self.sumErr+self.kd*derTerm
		self.lastErr=e
		return u
		
class KPID():#PID using kalman filter to estimate error derivative
	def __init__(self,kp=1,ki=1,kd=1,clip=None):
		self.kp=kp
		self.ki=ki
		self.kd=kd
		self.state=np.array([[0],[0]])
		self.A=lambda dt:np.eye(2)+np.array([[0,dt],[0,0]])
		self.Q=np.eye(2)
		self.P=np.eye(2)
		self.R=.01#measurement noise
		self.highClip=None
		self.lowClip=None
		self.saturateIntegral = False
		if clip !=None:
			self.saturateIntegral =True
			self.highClip=clip[1]
			self.lowClip=clip[0]
		self.sumErr=0
		
	def run(self,e,dt):
		H=np.array([[1,0]])
		self.state=np.matmul(self.A(dt),self.state)
		PHT=np.matmul(self.P,H.T)#PH'
		INV=np.linalg.inv(np.matmul(H,PHT)+self.R)#(HPH'+R)^-1
		K=np.matmul(PHT,INV) #PH'(HPH'+R)^-1
		self.state=self.state+K*e#x+K(z-Hx)
		self.P= np.matmul(np.eye(2)-np.matmul(K,H),self.P)#(I-KH)P
		self.sumErr += e*dt
		if self.saturateIntegral and self.lowClip != None and self.highClip != None:
			self.sumErr= min(max(self.sumErr,self.lowClip),self.highClip)
		derTerm=self.state[1,0]
		u = self.kp*self.state[0,0]+self.ki*self.sumErr+self.kd*derTerm
		return u	
					
class Swimming():
	def __init__(self,bias,amp,freq, wave = "square"):
		self.bias = bias
		self.amp = amp
		self.freq = freq
		self.wave = wave
		self.running = False
		self.socket = None
		
	def __str__(self):
		return "Swimming params:\n\tbias: {0},\n\tamplitude: {1}\n\tfrequency: {2}\n\twave type: {3}".format(self.bias,self.amp,self.freq,self.wave)
		
	def _handler(self):
		self.running = True
		try:
			switch ,last_angle= 1,readServoPos(self.socket)
		except:
			switch ,last_angle= 1,0
		t0=time.time()
		t_last= t0
		while self.running:
			t=time.time()
			if self.wave == "square":
				switch=-switch if t-t_last>1/max(self.freq,.05) else switch
				angle= self.bias + switch*self.amp
				if (t-t_last > 1/max(self.freq,.05) and abs(angle-last_angle)>.75) or (angle==self.bias and abs(angle-last_angle)>.75):
					t_last=t 
					last_angle=angle
					self.socket.send(str.encode("S,servo,"+str(int(angle))+",\n","utf-8"))
			if self.wave == "sin":
				angle = int(self.bias+self.amp*np.sin(2*np.pi*self.freq*(t-t0)%(2*np.pi)))
				if abs(angle-last_angle)>0.75:
					last_angle=angle
					self.socket.send(str.encode("S,servo,"+str(angle)+",\n","utf-8"))
			time.sleep(.02)#no faster than 50 hz
				
	def run(self,sock):
		if sock==None:
			return
		self.socket=sock
		self.running = True
		self.thread = threading.Thread(target=self._handler)
		self.thread.start()
		
	def stop(self):
		self.running = False
	def __del__(self):
		self.running = False



def connectToServer(server_address):
		sock =socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
		try:
			#sock.bind((host,port))
			sock.connect("\0"+server_address)
		except socket.error as msg:
				print(msg)
		return sock

if __name__ ==	"__main__":
	
	sock=connectToServer('./I2C_NODE')			
	#tail = Swimming(50,70,.75)
	tail = Swimming(50,70,.25,"sin")
	tail.run(sock)

