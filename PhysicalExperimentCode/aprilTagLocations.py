import numpy as np
from numpy import sin,cos
# yaw equation integration dyaw= (w_2*sin(roll)+w_3*cos(roll))/cos(pitch)
def rot2eul2(R):
	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if  not singular:
		x = np.arctan2(R[2,1] , R[2,2])*180.0/np.pi
		y = np.arctan2(-R[2,0], sy)*180.0/np.pi
		z = np.arctan2(R[1,0], R[0,0])*180.0/np.pi
		return (x,y,z)

def rot2eul(R):
	sy = np.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])
	singular = sy < 1e-6
	if  not singular:
		x = np.arctan2(R[2,1] , R[2,2])#*180.0/np.pi #roll
		y = np.arctan2(-R[2,0], sy)#*180.0/np.pi #pitch
		z = np.arctan2(R[1,0], R[0,0])#*180.0/np.pi #yaw
		return (x,y,z)
		
def rotX(gamma):
    return np.array([[1, 0,0],[0, cos(gamma), -sin(gamma)],[0 , sin(gamma), cos(gamma)]])

def rotY(beta):
    return np.array([[cos(beta),0,sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]])

def rotZ(alpha):
    return np.array([[cos(alpha),-sin(alpha),0],[sin(alpha),cos(alpha),0],[0,0,1]])
	
def zyxRot(alpha,beta,gamma):#roll, pitch, yaw
		return np.matmul(np.matmul(rotZ(gamma),rotY(beta)),rotX(alpha))

def eul2rotm(eul):
	alpha,beta,gamma=eul
	return zyxRot(alpha,beta,gamma)
	
def RpToTf(R,p):
	tf = np.eye(4)
	tf[0:3,0:3]=R
	tf[:3,3]=p[:,0]
	tf[:3,3][np.abs(tf[:3,3])<1e-3]=0
	tf[np.abs(tf)<1e-9]=0
	return tf	
	
def RpToInvTf(R,p):
	tfr = np.eye(4)
	tft = np.eye(4)
	tfr[0:3,0:3]=R.transpose()
	tft[:3,3]=-p[:,0]
	tf=np.matmul(tfr,tft)
	tf[:3,3][np.abs(tf[:3,3])<1e-3]=0
	tf[np.abs(tf)<1e-9]=0
	return tf
	
def vecToTf(vec):
	tf = np.eye(4)
	eul = np.deg2rad(vec[3:])
	R=eul2rotm(eul)
	tf[0:3,0:3]=R
	pos = vec[:3]
	tf[:3,3]=pos
	tf[:3,3][np.abs(tf[:3,3])<1e-3]=0
	tf[np.abs(tf)<1e-9]=0
	return tf
	
def tfToVec(tf):
	R=tf[0:3,0:3]
	eul=rot2eul(R)
	pos=tf[:3,3]
	return np.array([pos[0],pos[1],pos[2],eul[0],eul[1],eul[2]])
	
def getCameraWorldPose(tagId,tag_pos_in_cam_frame,tag_R_in_cam_frame):
    #takes tag id, april detection translation, and april detection rotation matrix and returns 
	#homogeneous transformation matrix for camera world pose
	tag_pose_world_frame=idMap[tagId]
	cam_pose_tag_frame=np.linalg.inv(RpToTf(tag_R_in_cam_frame,tag_pos_in_cam_frame))
	return np.matmul(tag_pose_world_frame,cam_pose_tag_frame)
	
#we have known pose (position and orientation) of the tags in the world coordinate system
#we measure the pose of the tag relative to the camera coordinate system
#to get world location of the camera, we need to 
#  1.) rotate relative pose into world coordinate system
#  2.) translate tag position by relative position
# orv
#  1.) get camera pose in tag coordinate frame
#  2.) transform position camera position from tag frame to world frame
# in terms of homogeneous transformation matrices, that means camera in world frame  = (camera in tag frame)*(tag in world frame)

#with depth and imu, to get world location of the camera,
idMap={}
tankPoses=[]
maxID=0
try:
	tankPoses=np.loadtxt("calibrationData/AprilTagTankLocations.csv",delimiter=',',skiprows=1)
	for id,x,y,z,roll,pitch,yaw in tankPoses:
		idMap[id]=vecToTf([x,y,z,roll,pitch,yaw]) #[x,y,z,roll,pitch,yaw] in world frame, Euler angles in degrees
		#print([x,y,z,roll,pitch,yaw]) #[x,y,z,roll,pitch,yaw] in world frame, Euler angles in degrees
		if id > maxID:
			maxID = id
except:
	pass
#print(tankPoses.shape)

#AprilTag Kalman Filter 
#state:x,y,z,yaw,vx,vy,vz,vyaw
Papril=0.5*np.eye(8)
Qapril=np.diag([0.05 ,0.05 ,0.05,np.deg2rad(3),0.1,0.1,0.1,np.deg2rad(1.5)])
Rmeas1april=1/100 #pressure sensor depth variance
#Rmeas2april=[.2,0.2,0.2,np.deg2rad(10)**2]#april tag x,y,z,yaw 
Rmeas2april=[1,1,1,np.deg2rad(10)**2]#april tag x,y,z,yaw 
Aapril=np.eye(8)
rdtapril=[0,1,2]
cdtapril=[4,5,6]
Bapril=np.array([[0],[0],[0],[1],[0],[0],[0],[0]])
GPSxynoise=.02**2
GPSyawnoise=np.deg2rad(3)**2
imu_in_camera_frame=vecToTf([0,0,-.2,-90,-90,0])
boundariesXY=[max(tankPoses[:,1]),min(tankPoses[:,1]),max(tankPoses[:,2]),min(tankPoses[:,2])]#xmin,xmax,ymin,ymax
windowLen=30#cfg.windowLen
window=np.ones((3,windowLen))*-1000#time,x,y
windowTime=30	
		
