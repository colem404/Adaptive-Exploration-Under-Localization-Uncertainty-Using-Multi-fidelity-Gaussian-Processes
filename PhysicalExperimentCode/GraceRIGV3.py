import numpy as np
import json
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import scipy.spatial.kdtree
import time
import ergodicKLDivergence as ekld

def angleWrap(angle,wrapVal):
	return (angle+wrapVal)%(2.0*wrapVal)-wrapVal 
	

class GraceAgent():
	def __init__(self,state=np.array([[0],[0],[0]])):
		self.state=state #planning state: x,y,z
		self.robotState=[]
		self.mfgp=None
		self.mfgp2=None
		self.fidLevs=[]
		self.sfgp=None
		self.sfgp2=None
		self.logDetPrior=None
		self.EID=None
		self.ergSigma=None
		self.fieldGrid=None
		self.CalcCost=self.calcPathErgodicity#self.calcPathInfo 
		self.trajCount=20
		self.legTypes=["Spiral","Glide","Swim","FlatDive"]
		self.legProbs=[1/4,1/4,1/4,1/4]
		self.numLegs=3
		self.spiralSpeed=.015
		self.vertGlideSpeed=.015
		self.flatDiveSpeed=.01
		self.swimSpeed=.05
		self.measRate=1 #rate at which measurements are taken
		self.maxDepth=1
		self.underWaterTimeLimit=60*5
		self.varianceRate=0
		self.minRadius=np.deg2rad(40)#lower tail angle for spiral
		self.maxRadius=np.deg2rad(90)#higher tail angle for spiral
		self.maxGlidePathAng=np.deg2rad(90)
		self.minGlidePathAng=np.deg2rad(30)
		self.SurfaceBySpiral=False
		self.FlatDiveEnergy=0.1
		self.GlideEnergy=0.15
		self.timeEnergy=0.005
		self.tailAmp=np.deg2rad(45)
		self.tailFreq=.75#hz
		self.tailEnergyScale=.5
		self.stopWatchTime=None
		self.stopWatchDuration=60
	
	def stopWatch(self):
		if self.stopWatchTime==None:
			self.stopWatchTime=time.time()
		else:
			return time.time()-self.stopWatchTime<self.stopWatchDuration
		return True
		
	def tailParamsFromSpeed(self,speed):
		return 0,0
		
	def SwimEnergy(self,t,f,a):
		wt=4*np.pi*f*t
		return 0.5*np.pi*a**2*f*(np.sin(wt)+wt)
		
	def Steer(self,x1, x2, delta):
		d=np.linalg.norm(x2-x1,2)
		if d==0:
			return x1
		xnew=x1+min(d,delta)*(x2-x1)/np.linalg.norm(x2-x1,2)
		return xnew
		
	def getStateDistance(self,state1,state2):
		d=np.sqrt((state1[0]-state2[0])**2+(state1[1]-state2[1])**2)
		#d=np.linalg.norm(x2-x1,2)
		return d
		
	def getpstate(self,state):
		return state#np.array([[state[0]],[state[1]]])
		
	def getCost(self,state1,state2):
		return np.linalg.norm(state2-state1,2)	
		
	def pruneStrategy(self):
		False
		
	def generateTraj(self,choices,distance):
		timeTaken,distanceTraveled,depth,prims=(0,0,0,[])
		cnt=0
		for c in choices:
			cnt+=1
			dz=0
			if c==self.legTypes[0]:#spiral
				d=np.random.rand()*self.maxDepth
				dz=d-depth
				r=self.minRadius+np.random.rand()*(self.maxRadius-self.minRadius)
				prim=(c,dz,r,np.sign(dz)*self.spiralSpeed)#params:change in depth,tail angle
			elif c==self.legTypes[1]:#glide
				gp=self.minGlidePathAng+np.random.rand()*(self.maxGlidePathAng-self.minGlidePathAng)
				d=np.random.rand()*self.maxDepth
				dz=d-depth				
				prim=(c,gp*np.sign(dz),dz,np.sign(dz)*self.vertGlideSpeed)#params:glide path,change in depth
			elif c==self.legTypes[2]:#swim
				d=np.random.rand()*(distance-distanceTraveled)
				prim=(c,d,self.swimSpeed)#params:distance
			elif c==self.legTypes[3]:#flat dive
				d=np.random.rand()*self.maxDepth
				dz=d-depth
				prim=(c,dz,np.sign(dz)*self.flatDiveSpeed)#params: chang in depth
			else:
				continue
			tt,dt=self.evaluateTraj([prim])
			if distanceTraveled+dt<distance:
				if cnt==len(choices):
					if c==self.legTypes[0]:
						prim=(c,-depth,r,-self.spiralSpeed)#return to surface
						depth+=-depth
						tt,dt=self.evaluateTraj([prim])
						prims.append(prim)
						prim=(self.legTypes[2],distance-distanceTraveled,self.swimSpeed)# and swim to destination
						tt2,dt2=self.evaluateTraj([prim])
						timeTaken+=tt+tt2
						distanceTraveled+=dt+dt2
						prims.append(prim)	
					elif c==self.legTypes[3]:
						prim=(c,-depth,-self.flatDiveSpeed)#return to surface
						depth+=-depth
						tt,dt=self.evaluateTraj([prim])
						prims.append(prim)
						prim=(self.legTypes[2],distance-distanceTraveled,self.swimSpeed)# and swim to destination
						tt2,dt2=self.evaluateTraj([prim])
						timeTaken+=tt+tt2
						distanceTraveled+=dt+dt2
						prims.append(prim)	
					elif c==self.legTypes[2]: #swim remaining distance
						prim=(c,distance-distanceTraveled,self.swimSpeed)
						tt,dt=self.evaluateTraj([prim])
						prims.append(prim)
						if depth>0:# spiral/flat dive to surface if underwater
							if self.SurfaceBySpiral:
								r=self.minRadius+np.random.rand()*(self.maxRadius-self.minRadius)
								prim=(self.legTypes[0],-depth,r,-self.spiralSpeed)
							else:
								prim=(self.legTypes[3],-depth,-self.flatDiveSpeed)
							depth+=-depth
							tt2,dt2=self.evaluateTraj([prim])
							tt=tt+tt2
							dt=dt+dt2
							prims.append(prim)
						timeTaken+=tt
						distanceTraveled+=dt
					elif c==self.legTypes[1]:#glide to the point or glide to surface with minimum glide angle
						gp=-max(abs(np.arctan2(depth,distance-distanceTraveled)),self.minGlidePathAng)
						dz=-depth#max((distance-distanceTraveled)/np.tan(gp),-depth)
						prim=(c,gp,dz,-self.vertGlideSpeed)
						tt,dt=self.evaluateTraj([prim])
						prims.append(prim)
						if distance>distanceTraveled+dt:
							prim=(self.legTypes[2],distance-distanceTraveled-dt,self.swimSpeed)
							tt2,dt2=self.evaluateTraj([prim])
							tt=tt+tt2
							dt=dt+dt2
							prims.append(prim)
							#print("glide fix 2")
						#print("glide fix 1",distance,distanceTraveled,gp,dt,dz,-depth)
						timeTaken+=tt
						distanceTraveled+=dt
						depth+=dz
						#print("glide too short! fixing final leg",depth,c,prims)
					#if depth>0.001 or abs(distanceTraveled-distance)>0.001:
					#	print("glide too short! fixing final leg",depth,c)
					#	print(depth,distance,distanceTraveled,timeTaken,prims)
					#	input()

				else:
					timeTaken+=tt
					distanceTraveled+=dt
					depth+=dz
					prims.append(prim)
			elif distanceTraveled+dt>=distance:#
				#print("too far!",c)
				if c==self.legTypes[1]:#glide to the point or glide to surface with minimum glide angle
					#print("glided too far! fixing final leg:",depth,prims,prim)
					if depth>0:
						gp=-max(abs(np.arctan2(depth,distance-distanceTraveled)),self.minGlidePathAng)
					else:
						gp=max(abs(np.arctan2(depth,distance-distanceTraveled)),self.minGlidePathAng)	
					dz=-depth#np.tan(gp)*(distance-distanceTraveled)
					prim=(c,gp,dz,-self.vertGlideSpeed)
					tt,dt=self.evaluateTraj([prim])
					prims.append(prim)
					if distance>distanceTraveled+dt:
							#print("not far enough:",depth,prims,prim)
							prim=(self.legTypes[2],distance-distanceTraveled-dt,self.swimSpeed)
							tt2,dt2=self.evaluateTraj([prim])
							#dz=0
							dt+=dt2
							tt+=tt2
							prims.append(prim)
					depth+=dz
					timeTaken+=tt
					distanceTraveled+=dt
				elif c==self.legTypes[2]: #shorten to swim remaining distance
					#print("swam too far! fixed final leg",prims)
					prim=(c,distance-distanceTraveled,self.swimSpeed)
					tt,dt=self.evaluateTraj([prim])
					prims.append(prim)
					timeTaken+=tt
					distanceTraveled+=dt
				#if abs(depth)>0.01  or abs(distanceTraveled-distance)>0.001:
				#	print("too far!",c)
				break
			else:
				timeTaken+=tt
				distanceTraveled+=dt
				depth+=dz
				prims.append(prim)
		if depth>0:# spiral/flat dive to surface if underwater
			if self.SurfaceBySpiral:
				r=self.minRadius+np.random.rand()*(self.maxRadius-self.minRadius)
				prim=(self.legTypes[0],-depth,r,self.spiralSpeed)
			else:
				prim=(self.legTypes[3],-depth,self.flatDiveSpeed)
			tt,dt=self.evaluateTraj([prim])
			depth+=-depth
			timeTaken+=tt
			distanceTraveled+=dt
			prims.append(prim)
		if abs(depth)>0.01  or abs(distanceTraveled-distance)>0.001:
			print("too far!",c)
			print(depth,distance,distanceTraveled,timeTaken,prims)
			input()
		return timeTaken,prims
		
		
	def evaluateTraj(self,prims,withTUW=False,withVar=False):
		timeTaken,distanceTraveled,budgetUsed,var=(0,0,0,0)
		tuws=[0]
		pnts=[]#distance,depth,time
		cnt=0
		depth=0
		uw=restart=False
		if withVar:
			pnts.append((distanceTraveled,depth,timeTaken,var))
		else:
			pnts.append((distanceTraveled,depth,timeTaken))
		for prim in prims:
			if prim[0]==self.legTypes[0]:#spiral
				_,dz,_,speed=prim
				timeTaken+=abs(dz/speed)
				tuws[cnt]+=abs(dz/speed)
				var+=self.varianceRate*abs(dz/speed)
				#distanceTraveled+=0
				depth=depth+dz
				budgetUsed+=self.GlideEnergy
			elif prim[0]==self.legTypes[1]:#glide
				_,gp,dz,speed=prim
				timeTaken+=abs(dz/speed)
				tuws[cnt]+=abs(dz/speed)
				var+=self.varianceRate*abs(dz/speed)
				distanceTraveled+=dz/np.tan(gp)
				depth=depth+dz
				budgetUsed+=self.GlideEnergy
			elif prim[0]==self.legTypes[2]:#swim
				_,dist,speed=prim
				timeTaken+=dist/speed
				tuws[cnt]+=uw*(dist/speed)
				var+=self.varianceRate*uw*(dist/speed)
				distanceTraveled+=dist
				budgetUsed+=self.SwimEnergy(dist/speed,self.tailAmp,self.tailFreq)*self.tailEnergyScale
				#depth=depth+0
			elif prim[0]==self.legTypes[3]:#flat dive
				_,dz,speed=prim
				timeTaken+=abs(dz/speed)
				tuws[cnt]+=abs(dz/speed)
				var+=self.varianceRate*abs(dz/speed)
				#distanceTraveled+=0
				depth=depth+dz
				budgetUsed+=self.FlatDiveEnergy
			if depth>0:# and not restart:
				uw=restart=True
			elif depth<=0.1 and restart:
				uw=restart=False
				tuws.append(0)
				cnt+=1
			if depth<=0:
				var = 0
			if withVar:
				pnts.append((distanceTraveled,depth,timeTaken,var))
			else:
				pnts.append((distanceTraveled,depth,timeTaken))
		budgetUsed+=self.timeEnergy*timeTaken
		if withTUW:
			return timeTaken,distanceTraveled,max(tuws),pnts,budgetUsed
		return timeTaken,distanceTraveled
		
	def edgePlanner(self,x1,x2,Env,timeConstraint=True):
		trajCount=self.trajCount
		legTypes=self.legTypes
		probs=self.legProbs
		numLegs=self.numLegs
		eList=[]
		for i in range(trajCount):
			if len(legTypes)!=len(probs):
				print("prob vector must match number of leg types")
				return 
			if x1.idx==x2.idx: #same node. column sample is only trajectory
				#legTypes=["Spiral","Glide","Swim","FlatDive"]
				probs=[self.SurfaceBySpiral*1,0,0,(not self.SurfaceBySpiral)*1] 
			choices=np.random.choice(legTypes,numLegs,p=probs)
			distance = np.linalg.norm(self.getpstate(x1.state)-self.getpstate(x2.state),2)
			tt,prims=self.generateTraj(choices,distance)
			#print(tt)
			tt2,dtrav,tuw,wpnts,bu=self.evaluateTraj(prims,withTUW=True)
			#print(tt2,dtrav,tuw)
			#print("wpnts",wpnts)
			#if wpnts[-1][1] >1e-9 or abs(distance-dtrav)>0.001:
			#	print("problem!\n",wpnts[-1][1],distance,dtrav,wpnts)
			#	print("nodes",x1.idx,x2.idx)
			#	print(prims)
			#	input()
			info=-np.inf
			if Env!=None:
				testPoins=self.edgePointsToTrajPoints(x1,x2,wpnts)
				info= sum([Env(np.array([x[0:3]])) for x in testPoins])
				#info= sum([Env(x[0:3]) for x in testPoins])
			time=tt
			uncertainty=0#change in uncertainty (variance)
			#budget=tt
			budget=bu
			e=(x1.idx,x2.idx,info,budget,time,uncertainty,tuple(prims))
			if not timeConstraint:
				eList.append(e)
			elif tuw<=self.underWaterTimeLimit:
				eList.append(e)
		return eList
		
	def gpInformationGain(self,trajPnts,K,sig_n):
		x=trajPnts[0]
		sig_y=K(x,x)
		I=np.log(1+sig_y/sig_n)
		X=np.array([x])
		for x in trajPnts[1:]:
			x=_np.array([x])
			K11=K(X,X)
			K12=K(x_,X)
			K21=K12.T
			K22=K(x_,x_)
			I_n=np.eye(K11.shape[0])
			sig_y=K22-np.matmul(np.matmul(K12,np.inv(K11+sig_n*I_n)),K21)
			I+=np.log(1+sig_y/sig_n)
			X=np.concatenate((X,x_))
		return I	
		
	def mfgpInformationGain(self,trajPnts,fids,models):
		x=trajPnts[0]
		sig_y=K(x,x)
		I=np.log(1+sig_y/sig_n)
		X=np.array([x])
		for x in trajPnts[1:]:
			x=_np.array([x])
			K=0
			sig_n=0
			I_n=np.eye(K.shape[0])
			K11=K(X,X)
			K12=K(x,X)
			K21=K12.T
			K22=K(x,x)
			sig_y=K22-np.matmul(np.matmul(K12,np.inv(K11+sig_n*I_n)),K21)
			I+=np.log(1+sig_y/sig_n)
			X=np.concatenate((X,x_))
		return I
		
	def edgePointsToTrajPoints(self,n_prev,n_next,pnts,t_off=0,withVar=False):
		ps = n_prev.state
		pf = n_next.state
		diff=pf-ps
		b= np.arctan2(diff[1,0],diff[0,0])
		ddt=np.array(pnts)#distance, depth,timeStamp array
		timePoints=np.arange(0,pnts[-1][2],1/self.measRate)+t_off
		timePoints.shape=(timePoints.shape[0],1)
		extdist=np.interp(timePoints,ddt[:,2]+t_off,ddt[:,0])
		extdepth=np.interp(timePoints,ddt[:,2]+t_off,ddt[:,1])
		if withVar:
			extVar=np.interp(timePoints,ddt[:,2]+t_off,ddt[:,3])
		#temp=ddt[0:ddt.shape[0],0:1]
		#pnts3D=np.concatenate((ps.T+np.zeros((ddt.shape[0],ps.shape[0])),ddt[0:ddt.shape[0],1:4]),axis=1)+temp*np.array([np.cos(b),np.sin(b),0,0])
		if withVar: #x,y,depth,timeStamp,variance array
			pnts3D=np.concatenate((ps.T+np.zeros((extdepth.shape[0],ps.shape[0])),extdepth,timePoints,extVar),axis=1)+extdist*np.array([np.cos(b),np.sin(b),0,0,0])			
		else:	#x,y,depth, timeStamp array
			pnts3D=np.concatenate((ps.T+np.zeros((extdepth.shape[0],ps.shape[0])),extdepth,timePoints),axis=1)+extdist*np.array([np.cos(b),np.sin(b),0,0])
		#return pnts3D.tolist() 
		return pnts3D #x,y,depth, timeStamp array
		
	def pathToTrajPoints(self,V,E,path,dense=False,t_off=0,withVar=False):
		densePoints=None
		if withVar:
			pnts3D=np.array([[],[],[],[],[]]).T
		else:
			pnts3D=np.array([[],[],[],[]]).T
		for data in path:
			idx1,idx2,edge_idx=data[0:3]
			edgeID=(idx1,idx2)
			#print(edgeID)
			edge=E[edgeID][edge_idx]
			idx1,idx2,info,budget,time,uncertainty,prims=edge
			ttrav,dtrav,tuw,wpnts,bu=self.evaluateTraj(prims,withTUW=True,withVar=withVar)
			if type(densePoints)==type(None) and dense:
				densePoints=self.edgePointsToTrajPoints(V[idx1],V[idx2],wpnts,t_off=t_off,withVar=withVar)
			elif dense:
				densePoints=np.concatenate((densePoints,self.edgePointsToTrajPoints(V[idx1],V[idx2],wpnts,t_off=t_off,withVar=withVar)))		
			ps = V[idx1].state
			pf = V[idx2].state
			diff=pf-ps
			b= np.arctan2(diff[1,0],diff[0,0])
			ddt=np.array(wpnts)#distance, depth,timeStamp array
			ddt[:,2]=ddt[:,2]+t_off
			if withVar:
				temp=np.concatenate((ps.T+np.zeros((ddt.shape[0],ps.shape[0])),ddt[:,1:4]),axis=1)+ddt[:,0:1]*np.array([np.cos(b),np.sin(b),0,0,0])
			else:
				temp=np.concatenate((ps.T+np.zeros((ddt.shape[0],ps.shape[0])),ddt[:,1:3]),axis=1)+ddt[:,0:1]*np.array([np.cos(b),np.sin(b),0,0])
			pnts3D=np.concatenate((pnts3D,temp))
			t_off+=wpnts[-1][-1]
		if dense:
			_,ind=np.unique(np.round(densePoints,4),axis=0,return_index=True)
			return densePoints[np.sort(ind),:]#np.unique(densePoints,axis=0)
		_,ind=np.unique(np.round(pnts3D,4),axis=0,return_index=True)
		return pnts3D[np.sort(ind),:]#np.unique(pnts3D,axis=0)
	
	def calcPathInfo(self,V,E,path,dense=False):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)
		fl=self.fidLevs
		l1=pnts[:,-1]<fl[0]
		l2=np.logical_and(pnts[:,-1]>fl[0], pnts[:,-1]<fl[1])
		l3=np.logical_and(pnts[:,-1]>fl[1], pnts[:,-1]<fl[2])
		X=[pnts[l3,:3],pnts[l2,:3],pnts[l1,:3]]
		
		#need to add pnts to appropriate fidelity
		#print(pnts)
		#input()
		I=self.mfgp.calcInformationGain(X)
		return I
	
	def calcPathInfoSF2(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense)
		X=pnts[1:,:3]
		if 0 in X.shape:
			return -np.inf
		sfgp=self.sfgp.copy()
		#_,Kprior= sfgp.predict(np.concatenate((sfgp.X,X)),full_cov=1)
		sig_n=sfgp.Gaussian_noise.variance[0]
		xtemp=pnts[0,:3]
		xtemp.shape=(1,3)
		#sfgp.set_XY(xtemp,np.array([[0]]))
		sfgp.set_XY(np.concatenate((sfgp.X,xtemp)),np.concatenate((sfgp.Y,np.array([[0]]))))
		_,sig_y=sfgp.predict(xtemp)
		I=np.log(1+sig_y[0,0]/sig_n)
		for i in range(X.shape[0]):
			xtemp=X[i,:3]
			xtemp.shape=(1,3)
			_,sig_y=sfgp.predict(xtemp)
			I+=np.log(1+sig_y[0,0]/sig_n)
			#sfgp.set_XY(X[:i+1,:3],np.zeros((i+1,1)))
			sfgp.set_XY(np.concatenate((sfgp.X,xtemp)),np.concatenate((sfgp.Y,np.array([[0]]))))
		#_,Kposterior= sfgp.predict(sfgp.X,full_cov=1)
		#print(I,0.5*(np.log(np.linalg.det(Kprior))-np.log(np.linalg.det(Kposterior))))
		return I
		
	def calcPathInfoSF3(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense)
		X=pnts[1:,:3]
		if 0 in X.shape:
			return -np.inf
		if self.sfgp2 == None:
			self.sfgp2=self.sfgp.copy()
		sfgp=self.sfgp2
		#_,Kprior= sfgp.predict(np.concatenate((sfgp.X,X)),full_cov=1)
		sig_n=sfgp.Gaussian_noise.variance[0]
		xtemp=pnts[0,:3]
		xtemp.shape=(1,3)
		#sfgp.set_XY(xtemp,np.array([[0]]))
		sfgp.set_XY(np.concatenate((self.sfgp.X,xtemp)),np.concatenate((self.sfgp.Y,np.array([[0]]))))
		_,sig_y=sfgp.predict(xtemp)
		I=np.log(1+sig_y[0,0]/sig_n)
		for i in range(X.shape[0]):
			xtemp=X[i,:3]
			xtemp.shape=(1,3)
			_,sig_y=sfgp.predict(xtemp)
			I+=np.log(1+sig_y[0,0]/sig_n)
			#sfgp.set_XY(X[:i+1,:3],np.zeros((i+1,1)))
			sfgp.set_XY(np.concatenate((sfgp.X,xtemp)),np.concatenate((sfgp.Y,np.array([[0]]))))
		#_,Kposterior= sfgp.predict(sfgp.X,full_cov=1)
		#print(I,0.5*(np.log(np.linalg.det(Kprior))-np.log(np.linalg.det(Kposterior))))
		return I
	
	def calcPathInfoSF4(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense)
		X=pnts[1:,:3]
		if 0 in X.shape:
			return -np.inf
		if self.sfgp2 == None:
			self.sfgp2=self.sfgp.copy()
		sfgp=self.sfgp2
		lx,ly,lz=sfgp.kern.lengthscale
		sig_n=sfgp.Gaussian_noise.variance[0]
		xtemp=pnts[0,:3]
		xtemp.shape=(1,3)
		allX=np.concatenate((self.sfgp.X,xtemp))
		tempX=allX[np.logical_and(allX[:,0]<3*lx,allX[:,1]<3*ly)]						   
		tempY=np.zeros((tempX.shape[0],1))
		sfgp.set_XY(np.concatenate((tempX,xtemp)),np.concatenate((tempY,np.array([[0]]))))
		_,sig_y=sfgp.predict(xtemp)
		I=np.log(1+sig_y[0,0]/sig_n)
		for i in range(X.shape[0]):
			xtemp=X[i,:3]
			xtemp.shape=(1,3)					
			allX=np.concatenate((allX,xtemp))
			tempX=allX.copy()
			if allX.shape[0]>100:
				tempX=allX[np.logical_and(allX[:,0]<3*lx,allX[:,1]<3*ly)] 
				tempX=allX if tempX.shape[0]==0 else tempX
			tempY=np.zeros((tempX.shape[0],1))
			sfgp.set_XY(tempX,tempY)
			_,sig_y=sfgp.predict(xtemp)
			if np.any(sig_y<0):
				print(sig_y)
			I+=np.log(1+sig_y[0,0]/sig_n)
		return I
		
	def calcPathInfoSF(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense)
		X=pnts[1:,:3]
		if 0 in X.shape:
			return -np.inf
		sfgp=self.sfgp.copy()
		lx,ly,lz=sfgp.kern.lengthscale
		sig_n=sfgp.Gaussian_noise.variance[0]
		xtemp=pnts[0,:3]
		xtemp.shape=(1,3)
		#sfgp.set_XY(xtemp,np.array([[0]]))
		allX=np.concatenate((sfgp.X,xtemp))
		#tempX=allX[np.argsort(np.linalg.norm(allX-xtemp,2,1))[:min(allX.shape[0],100)]]
		tempX=allX[np.logical_and(allX[:,0]<3*lx,allX[:,1]<3*ly)]						   
		tempY=np.zeros((tempX.shape[0],1))
		sfgp.set_XY(np.concatenate((sfgp.X,xtemp)),np.concatenate((sfgp.Y,np.array([[0]]))))
		_,sig_y=sfgp.predict(xtemp)
		I=np.log(1+sig_y[0,0]/sig_n)
		for i in range(X.shape[0]):
			xtemp=X[i,:3]
			xtemp.shape=(1,3)
							  
								
			#sfgp.set_XY(X[:i+1,:3],np.zeros((i+1,1)))	
			allX=np.concatenate((allX,xtemp))
			tempX=allX.copy()
			if allX.shape[0]>100:
				#tempX=allX[np.argsort(np.linalg.norm(allX-xtemp,2,1))[:min(allX.shape[0],100)]]
				tempX=allX[np.logical_and(allX[:,0]<3*lx,allX[:,1]<3*ly)] 
				tempX=allX if tempX.shape[0]==0 else tempX
				#print(allX.shape,tempX.shape)
			tempY=np.zeros((tempX.shape[0],1))
			sfgp.set_XY(tempX,tempY)
			_,sig_y=sfgp.predict(xtemp)
			if np.any(sig_y<0):
				print(sig_y)
			I+=np.log(1+sig_y[0,0]/sig_n)
		return I
	
	def calcPathInfoSFBatch(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)#row->x,y,z,t,sig
		X=pnts[1:,:3]
		Xpred=self.fieldGrid
		n=Xpred.shape[0]
		#c=(2*np.pi*np.exp(1))**n
		if self.sfgp2 == None:
			self.sfgp2=self.sfgp.copy()
		sfgp=self.sfgp2
		
		if self.logDetPrior == None:
			sfgp.set_XY(self.sfgp.X,self.sfgp.Y)
			_,Kprior=sfgp.predict(Xpred,full_cov=1)
			det=np.linalg.det(Kprior)
			if det==0:
				det=np.linalg.det(np.eye(n)*(sfgp.kern.variance[0]+sfgp.Gaussian_noise.variance[0]))
			#print(det)
			self.logDetPrior=np.log(det)
			#print(self.logDetPrior)
		sfgp.set_XY(np.concatenate((sfgp.X,X)),np.concatenate((sfgp.Y,np.zeros((X.shape[0],1)))))
		_,Kposterior=sfgp.predict(Xpred,full_cov=1)
		det=np.linalg.det(Kposterior)
		logdet=0 if det==0 else np.log(det)
		I=max(0.5*(self.logDetPrior-logdet),0)
		if np.isinf(I):
			I=0
		return I
	
	def calculatePathInfoEmuBatch(self,V,E,path,dense=False):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)#row->x,y,z,t,sig
		
		fl=self.fidLevs
		l1=pnts[0:,-1:]<fl[0]#highest fidelity
		l2=np.logical_and(pnts[0:,-1:]>fl[0], pnts[0:,-1:]<fl[1])
		l3=np.logical_and(pnts[0:,-1:]>fl[1], pnts[0:,-1:]<fl[2])#lowest fidelity
		X=np.concatenate((pnts[0:,:3],l1*2+l2*1+l3*0),axis=1)
		Xpred=np.hstack((self.fieldGrid,2*np.ones((self.fieldGrid.shape[0],1))))
		if self.mfgp2 == None:
			self.mfgp2=self.mfgp.copy()
		mfgp=self.mfgp2
		mfgp.set_data(self.mfgp.X,self.mfgp.Y)
		if self.logDetPrior == None:
			Kprior=mfgp.predict_covariance(Xpred)
			self.logDetPrior=np.log(np.linalg.det(Kprior))
		mfgp.set_data(np.concatenate((mfgp.X,X)),np.concatenate((mfgp.Y,np.zeros((X.shape[0],1)))))
		Kposterior=mfgp.predict_covariance(Xpred)
		I=0.5*(self.logDetPrior-np.log(np.linalg.det(Kposterior)))
		return I
	
	def calculatePathInfoEmu2(self,V,E,path,dense=False):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)#row->x,y,z,t,sig
		
		fl=self.fidLevs
		l1=pnts[0:,-1:]<fl[0]#highest fidelity
		l2=np.logical_and(pnts[0:,-1:]>fl[0], pnts[0:,-1:]<fl[1])
		l3=np.logical_and(pnts[0:,-1:]>fl[1], pnts[0:,-1:]<fl[2])#lowest fidelity
		X=np.concatenate((pnts[0:,:3],l1*2+l2*1+l3*0),axis=1)
		Xpred=np.concatenate((pnts[0:,:3],l1*0+2),axis=1)
		mfgp=self.mfgp
		Kprior=mfgp.gpy_model.kern.K(Xpred)
		oriX=mfgp.X
		oriY=mfgp.Y
		hyps=mfgp.gpy_model.param_array
		mfgp.set_data(np.concatenate((mfgp.X,X)),np.concatenate((mfgp.Y,np.zeros((X.shape[0],1)))))
		Kposterior=mfgp.predict_covariance(Xpred)
		I=0.5*(np.log(np.linalg.det(Kprior))-np.log(np.linalg.det(Kposterior)))
		mfgp.set_data(oriX,oriY)
		return I
		
		
	def calculatePathInfoEmu(self,V,E,path,dense=False):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)#row->x,y,z,t,sig
		
		fl=self.fidLevs
		l1=pnts[0:,-1:]<fl[0]#highest fidelity
		l2=np.logical_and(pnts[0:,-1:]>fl[0], pnts[0:,-1:]<fl[1])
		#l3=np.logical_and(pnts[0:,-1:]>fl[1], pnts[0:,-1:]<fl[2])#lowest fidelity
		l3=pnts[0:,-1:]>fl[1]#lowest fidelity
		X=np.concatenate((pnts[0:,:3],l1*2+l2*1+l3*0),axis=1)
		
		mfgp=self.mfgp
		lx,ly,lz=mfgp.gpy_model.kern.rbf.lengthscale
		oriX=mfgp.X
		oriY=mfgp.Y
		allX=oriX.copy()
		hyps=mfgp.gpy_model.param_array
		l1noise=hyps[-1]#noise variance
		#l2noise=hyps[-2]#noise variance
		#l3noise=hyps[-3]#noise variance
		sig_n=l1noise
		I=0
		for i in range(X.shape[0]):
			xtemp=X[i:i+1,0:]
			tempX=allX.copy()
			allX=np.concatenate((allX,xtemp))
			if allX.shape[0]>100:
				#tempX=allX[np.argsort(np.linalg.norm(allX[:,:3]-xtemp[:,:3],2,1))[:min(allX.shape[0],100)]]
				tempX=allX[np.logical_and(allX[:,0]<5*lx,allX[:,1]<5*ly)]
				#print(allX.shape,tempX.shape)
			mfgp.set_data(tempX,np.zeros((tempX.shape[0],1)))
			_,sig_y=mfgp.predict(np.array([xtemp.flatten()[:3].tolist()+[0]]))
			if np.any(sig_y<0):
				print(sig_y)
			I+=np.log(1+sig_y[0,0]/sig_n)
									
			#mfgp.set_data(np.concatenate((mfgp.X,xtemp)),np.concatenate((mfgp.Y,np.array([[0]]))))
		mfgp.set_data(oriX,oriY)
		return I
		
	#def calculatePathVarRed(self,V,E,path,dense=False):
	#	pnts=self.pathToTrajPoints(V,E,path,dense=dense,withVar=True)#row->x,y,z,t,sig
	#	fl=self.fidLevs
	#	l1=pnts[0:,-1:]<fl[0]#highest fidelity
	#	l2=np.logical_and(pnts[0:,-1:]>fl[0], pnts[0:,-1:]<fl[1])
	#	l3=np.logical_and(pnts[0:,-1:]>fl[1], pnts[0:,-1:]<fl[2])#lowest fidelity
	#	X=np.concatenate((pnts[0:,:3],l1*2+l2*1+l3*0),axis=1)
	#	mfgp=self.mfgp
	#	oriX=mfgp.X
	#	oriY=mfgp.Y
	#	hyps=mfgp.gpy_model.param_array
	#	l1noise=hyps[-1]#noise variance
	#	l2noise=hyps[-2]#noise variance
	#	l3noise=hyps[-3]#noise variance	
	#	I=np.sum(mfgp.calculate_variance_reduction(refSet,X))
	#	return I
	def calcPathErgodicity(self,V,E,path,dense=True):
		pnts=self.pathToTrajPoints(V,E,path,dense=dense)
		X=pnts[:,:3]
		t=pnts[:,3:4]
		Sigma= self.ergSigma if type(self.ergSigma)!=type(None) else 0.25*np.eye(3)
		fieldGrid=self.fieldGrid
		q=ekld.computeTrajectoryIntegrand(t,X,fieldGrid,Sigma)
		if np.any(q==0):
			q=q+min(min(q[q>0]),1e-15)
		p=self.EID.copy()
		if np.any(p==0):
			p=p+min(min(p[p>0]),1e-15)
		#ergCost=ekld.ergodicDivergence(1e-15+p,1e-15+q)
		#ergCost=ekld.ergodicDivergence(p,q)
		ergCost=ekld.ergodicDivergence(q,p)
		return -ergCost
		
	def CalcMaxTimeMinEnergy(self,V,E,path,dense=False):
		data = path[-1]
		idx1,idx2,edge_idx=data[0:3]
		edgeID=(idx1,idx2)
		edge=E[edgeID][edge_idx]
		idx1,idx2,info,budget,time,uncertainty,prims=edge
		return time/budget
	
class Geometric3DAgent():
	def __init__(self,state=np.array([[0],[0],[0]])):
		self.state=state #x,y,yaw,energy
		
	def Steer(self,x1, x2, delta):
		#xe,ye,ze=(x2-x1)[:,0].tolist()
		#dh=np.sqrt(xe**2+ye**2)#horizontal plane distance
		#d=np.sqrt(xe**2+ye**2+ze**2)#distance
		#b=np.arctan2(ye,xe)#bearing
		#a=np.arctan2(ze,dh)#elevation angle
		#xnew=x1+min(d,delta)*np.array([[np.sin(b)*np.sin(a)],[np.cos(b)*np.sin(a)],[np.cos(a)]])	
		xnew=x1+min(d,delta)*(x2-x1)/np.linalg.norm(x2-x1,2)
		return xnew
		
	def getStateDistance(self,state1,state2):
		d=np.sqrt((state1[0]-state2[0])**2+(state1[1]-state2[1])**2+(state1[2]-state2[2])**2)
		return d
		
	def getpstate(self,state):
		return state#np.array([[state[0]],[state[1]],[state[2]]])
		
	def getCost(self,state1,state2):
		return 0
	
class GeometricNDAgent():
	def __init__(self,state=np.array([[0],[0]])):
		self.state=state #x,y,yaw,energy
		
	def Steer(self,x1, x2, delta):	
		xnew=x1+min(d,delta)*(x2-x1)/np.linalg.norm(x2-x1,2)
		return xnew
		
	def getStateDistance(self,state1,state2):
		d=np.linalg.norm(x2-x1,2)
		return d
		
	def getpstate(self,state):
		return state#np.array([[state[0]],[state[1]]])
		
	def getCost(self,state1,state2):
		return np.linalg.norm(state2-state1,2)	
		
	def pruneStrategy(self):
		pass	
		
class Geometric2DAgent():
	def __init__(self,state=np.array([[0],[0]])):
		self.state=state #x,y,yaw,energy
		self.trajCount=3
		self.speed=1
		
	def Steer(self,x1, x2, delta):
		#xe,ye=(x2-x1)[:,0].tolist()
		#d=np.sqrt(xe**2+ye**2)#distance
		#b=np.arctan2(ye,xe)#bearing
		#xnew=x1+min(d,delta)*np.array([[np.cos(b)],[np.sin(b)]])
		xnew=x1+min(d,delta)*(x2-x1)/np.linalg.norm(x2-x1,2)			
		return xnew
		
	def getStateDistance(self,state1,state2):
		d=np.sqrt((state1[0]-state2[0])**2+(state1[1]-state2[1])**2)
		return d
		
	def getpstate(self,state):
		return state#np.array([[state[0]],[state[1]]])
		
	def getCost(self,state1,state2):
		return np.linalg.norm(state2-state1,2)	
		
	def pruneStrategy(self):
		False
		
	def edgePlanner(self,x1,x2):
		trajCount=self.trajCount
		eList=[]
		for i in range(trajCount):
			distance = np.linalg.norm(self.getpstate(x1.state)-self.getpstate(x2.state),2)
			info=0
			time=distance/min(np.random.rand()+.1,1)#distance/self.speed
			uncertainty=0#change in uncertainty
			e=(x1.idx,x2.idx,time,distance,info,uncertainty)
			eList.append(e)
		return eList
	
	
	
class Node:
	def __init__(self,x,cost=0.0,info=0.0,uncertainty=0.0,neigbors = {}):
		self.idx=0
		self.neigbors = neigbors #startState,endState,cost,info
		self.state=x
		self.info=-np.inf
		#self.cost=0
		self.minPathCost=-np.inf
		self.maxPathCost=-np.inf
		self.path=[]
		self.minBudgetPath=[]
		self.maxBudgetPath=[]
		self.pathList=[]#list of tuples: (start node,end node,edge index,time,budget,info)
		
		
	def sortByIDXPathList(self,pathList,sortIdx=[0],rev=True):
		pathList.sort(reverse=rev,key=lambda x:[x[i] for i in sortIdx])
		#sort(reverse=True,key=lambda x:x[3])
		#sort(reverse=True,key=lambda x:(x[2],x[3]))
	
	def __str__(self):
		return "Node {0}: min budget cost={1},\n\tstate={2},\n\t max info={3}\n\tnum paths={4}".format(self.idx,self.minPathCost,self.state,self.info,len(self.pathList))
		
	def equal(self,cNode):
		return False
		
class Edge:
	def __init__(self,x1,x2,cost=0.0,info=0.0,uncertainty=0.0,neigbors = {}):
		self.cost=cost
		self.info=info
		self.id=(x1.idx,x2.idx)
		self.uncertainty=uncertainty#change in uncertainty
		self.time=0
		self.distance = 0
			
	def __str__(self):
		return "Node {0}: cost={1},\n\tstate={2},\n\tinfo={3}".format(self.idx,self.cost,self.state,self.info)
		
	def equal(self,cEdge):
		return self.cost==cEdge.cost and self.uncertaint==cEdge.uncertainty and self.info==cEdge.info
		
	def compare(self,cEdge):
		if self.equal(cEdge):
			return 0
		elif self.cost<cEdge.cost and self.uncertaint<cEdge.uncertainty and self.info>cEdge.info:
			return 1 # greater/better
		elif self.cost>cEdge.cost and self.uncertaint>cEdge.uncertainty and self.info<cEdge.info:
			return -1 # less/worse
		#returns None if not greater than, lessthan, or equal to



	
class RIG:
	''' delta - step size
		B - resource budget
		WS - workspace (making this a nD rectangle defined  by a 2xn matrix defining bounds for simplicity)
			Example for 3D WS = np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax]])
		FS - free space (for now, we will make this simply equal to free space i.e. no obstacles)
		xstart - start configuration
		R - near radius
	'''
	def __init__(self,delta,B,WS,FS,Env,R,agent=None):
		self.delta=delta
		self.B=B
		self.R=R
		self.WS=WS
		self.FS=FS
		self.Env=Env #objective to maximize
		self.agent=agent
		self.animate=False
		self.animateNewEdge=False
		self.animationSleep=0.01
		self.debugMode=False
		self.ObjectList=[]
		self.ModularCost=False
		self.budgetCutoff=.9
		
		self.bestPath=(0,-np.inf,None,None)
		self.maxIter = 20#00
		self.curIter = 0
		# customizable functions
		self.terminalCond=self.defaultTerminalCond
		self.Prune=self.defaultPruneStrategy
		self.NoCollision=self.defaultNoCollision
		self.Sample=self.defaultSample
		self.EvaluateCost=self.secondaryCost
		if agent==None:
			self.steer=self.defaultSteer
		else:
			self.Steer=agent.Steer
		#Initialize cost, information, starting node, node list, edge list, and graph
		self.V={}
		self.Vidx=set()#index set for nodes in V
		self.Vc=set()
		self.E={}#set of edges
	
	def EvaluateCost(x1,x2):
		return 0
		
	def defaultSteer(self,x1, x2, delta):
		return x1

	def Nearest(self,xsamp,V,R):#V is the set of all nodes minus the set of closed nodes
		#R disc radius for closest 
		#print(V)
		#print(self.V)
		#for idx in V:
		#	print(self.V[idx].state)
		#	print(self.agent.getpstate(self.V[idx].state))
		#	print(self.agent.getpstate(self.V[idx].state)-xsamp)
		V=list(V)
		#dlist=[np.linalg.norm(self.agent.getpstate(self.V[idx].state)-xsamp,2) for idx in V]
		dlist=[(R-np.linalg.norm(self.agent.getpstate(self.V[idx].state)-xsamp,2))**2 for idx in V]
		
		#print(type(dlist),dlist)
		nidx=V[dlist.index(min(dlist))]
		return self.V[nidx]
	
	def Near(self,x1,V,R,withNearest=False):#V is the set of all nodes minus the set of closed nodes
		nlist=[]
		min_idx=-1
		min_d=max([self.SameNodeDistance,R])
		for idx in V:
			#print(self.agent.getpstate(self.V[idx].state))
			#print(self.agent.getpstate(x1))
			d=self.agent.getpstate(self.V[idx].state)-self.agent.getpstate(x1)
			#print(d)
			if R>=np.linalg.norm(d,2): #(R-np.linalg.norm(d,2))**2<0.1
				nlist.append(self.V[idx]) 
			if min_d>=np.linalg.norm(d,2):
				min_idx=idx
				min_d=np.linalg.norm(d,2)
		if withNearest:
			return min_idx,nlist
		return nlist
		
	def createNode(self,prevNode,xnew,Vidx):
		#Calculate new information and cost
		newNode=Node(xnew)
		#newNode.info=self.Information(prevNode.info,self.agent.getpstate(xnew).T,self.Env)
		#newNode.cost=prevNode.cost+self.EvaluateCost(prevNode.state,xnew)
		newNode.idx=max(Vidx)+1
		newNode.path=prevNode.path.copy()
		newNode.path.append(newNode.idx)
		newNode.pathList=prevNode.pathList.copy()
		newEdge=(prevNode.idx,newNode.idx)
		return self.Prune(newNode),newNode,newEdge

	
		
	def defaultSample(self,WS):
		# retruns a sample point
		s=np.diff(WS)#get scale of dimension
		lb=WS[:,0]
		lb.shape=s.shape
		rs=lb+s*np.random.random(s.shape)
		#print(rs,rs.shape)
		return rs
	
	def defaultTerminalCond(self):
		''' by default stop after set number of iterations
			but can be overwriten'''
		self.curIter+=1
		return self.maxIter>self.curIter
	
	def	defaultPruneStrategy(self,n_new):
		#for idx in self.V:
		#	d=self.agent.getpstate(self.V[idx].state)-self.agent.getpstate(n_new.state)
		#	if np.linalg.norm(d)<0.05:
		#		return True
		return False
		
	def	defaultNoCollision(self,x1,x2,FS):
		pointInFS=((self.agent.getpstate(x2)-FS)>=0)[:,0].all() and ((FS-self.agent.getpstate(x2))>=0)[:,1].all()
		return pointInFS
	
	def Information(self,Inear,x,Env):
		return Inear+self.Env(x)
		
	def primaryCost(self,x1,x2):
		return np.linalg.norm(x1-x2,2)
		
	def secondaryCost(self,x1,x2):
		return 1#np.linalg.norm(x1-x2,2)
		
	def node_locs(self):#location of all nodes
		return [(idx,self.V[idx].state) for idx in self.V]
		
	def node_loc_dict(self,save=False,fname="graphNodes.txt"):#location of all nodes
		temp={}
		for idx in self.V:
			temp[idx]=self.V[idx].state.tolist()
		if save:
			with open(fname,'w') as con_file:
				con_file.write(json.dumps(temp))
		return temp
	
	def edge_dict(self,save=False,fname="graphNodes.txt"):#location of all nodes
		temp={}
		for idx in self.E:
			temp[str(idx)]=self.E[idx]
		if save:
			with open(fname,'w') as con_file:
				con_file.write(json.dumps(temp))
		return temp

	def load_graph(self,edgeFile,NodeFile):
		ef=open(edgeFile)
		Edges=json.load(ef)
		nf=open(NodeFile)
		Nodes=json.load(nf)
		for k in Edges.keys():
			startnode,endnode=k.replace("(","").replace(")","").split(",")
			edgeID=(int(startnode),int(endnode))
			self.E[edgeID]=Edges[k]
		for k in Nodes.keys():
			self.V[int(k)]=np.array(Nodes[k])
			self.Vidx.add(int(k))
		
	def draw_path(self,V,pathList):
		x=[]
		y=[]
		for idx in pathList:
			#print(idx)
			temp= V[idx].state
			xi,yi=[temp[0],temp[1]]
			x.append(xi)
			y.append(yi)
		plt.gcf()
		plt.plot(x,y, marker="o",color="red")
		plt.draw()
	
	def drawBestPath(self):
		pathBudget,pathInfo,bnode_idx,bpath_idx=self.bestPath
		if bpath_idx != None:
			self.draw_2D_path_projection(self.V,self.V[bnode_idx].pathList[bpath_idx])
			
	def draw_2D_path_projection(self,V,pathList):
		x=[]
		y=[]
		temp= V[0].state
		xi,yi=[temp[0],temp[1]]
		x.append(xi)
		y.append(yi)
		for info in pathList:
			#print(idx)
			#print(idx)
			idx=info[1]
			temp= V[idx].state
			xi,yi=[temp[0],temp[1]]
			x.append(xi)
			y.append(yi)
		plt.gcf()
		plt.plot(x,y, marker="o",color="black")
		plt.draw()
		
	def draw_3D_path(self,V,E,path,dense=True):
		x=[]
		y=[]
		z=[]
		xnodes=[]
		ynodes=[]
		znodes=[]
		dx,dy,dz=([],[],[])
		densePoints=None
		for data in path:
			idx1,idx2,edge_idx=data[0:3]
			edgeID=(idx1,idx2)
			#print(edgeID)
			edge=E[edgeID][edge_idx]
			idx1,idx2,info,budget,time,uncertainty,prims=edge
			ttrav,dtrav,tuw,wpnts,bu=self.agent.evaluateTraj(prims,withTUW=True)
			if type(densePoints)==type(None) and dense:
				densePoints=self.agent.edgePointsToTrajPoints(V[idx1],V[idx2],wpnts)
			elif dense:
				densePoints=np.concatenate((densePoints,self.agent.edgePointsToTrajPoints(V[idx1],V[idx2],wpnts)))
			ps = V[idx1].state
			pf = V[idx2].state
			diff=pf-ps
			b= np.arctan2(diff[1,0],diff[0,0])
			xnodes.append(ps[0,0])
			ynodes.append(ps[1,0])
			znodes.append(0)
			for pt in wpnts:
				
				x.append(ps[0,0]+pt[0]*np.cos(b))
				y.append(ps[1,0]+pt[0]*np.sin(b))
				z.append(pt[1])
		#print(np.unique(np.array([x,y,z]).T,axis=0)-self.agent.pathToTrajPoints(V,E,path,dense=False)[:,0:-1])	
		#print(self.agent.pathToTrajPoints(V,E,path,dense=False).shape)	
		#print(np.unique(np.array([x,y,z]).T,axis=0).shape)	
		plt.figure().add_subplot(projection='3d')
		plt.plot(x,y,z, marker="^",color="green")
		if dense:
			plt.plot(densePoints[:,0],densePoints[:,1],densePoints[:,2], marker="o",color="blue")
		#plt.plot(xnodes,ynodes,znodes, marker="o",color="blue")
		#plt.gca()
		xedges=[]
		yedges=[]
		zedges=[]
		for idx in E.keys():
			#print(idx)
			temp= V[idx[0]].state
			x1,y1=[temp[0],temp[1]]
			temp= V[idx[1]].state
			x2,y2= [temp[0],temp[1]]
			xnodes.append(x1)
			ynodes.append(y1)
			znodes.append(0)
			xnodes.append(x2)
			ynodes.append(y2)
			znodes.append(0)
			#plt.plot([x1,x2], [y1,y2],[0,0], marker="o")#,color="blue")
		plt.plot(xedges, yedges,zedges, marker="o")#,color="blue")
		plt.gcf().canvas.mpl_connect(
			'key_release_event',
			lambda event: [exit(0) if event.key == 'escape' else None])


		if self.WS is not None:
			xmin,xmax=self.WS[0,:]
			ymin,ymax=self.WS[1,:]
			plt.plot([xmin, xmax,
					  xmax, xmin,
					  xmin],
					 [ymin, ymin,
					  ymax, ymax,
					  ymin],[0,0,0,0,0],
					 "-k")
		
		
		plt.draw()
		#plt.show()
		plt.grid(True)
		plt.pause(self.animationSleep)
		
	def draw_graph(self,V,E,WS=None,rnd=None):
		plt.clf()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect(
			'key_release_event',
			lambda event: [exit(0) if event.key == 'escape' else None])
		if rnd is not None:
			rx,ry=rnd.tolist()
			plt.plot(rx, ry, "^k")
		    #if self.robot_radius > 0.0:
		     #   self.plot_circle(rx, ry, 2, '-r')
		for idx in E:
			#print(idx)
			temp= V[idx[0]].state
			x,y=[temp[0],temp[1]]
			temp= V[idx[1]].state
			x2,y2= [temp[0],temp[1]]
			plt.plot([x,x2], [y,y2], marker="o",color="blue")

		#for (ox, oy, size) in self.obstacle_list:
		  #  self.plot_circle(ox, oy, size)

		if WS is not None:
			xmin,xmax=WS[0,:]
			ymin,ymax=WS[1,:]
			plt.plot([xmin, xmax,
					  xmax, xmin,
					  xmin],
					 [ymin, ymin,
					  ymax, ymax,
					  ymin],
					 "-k")

		#plt.plot(self.start.x, self.start.y, "xr")
		#plt.plot(self.end.x, self.end.y, "xr")
		plt.axis("equal")
		#plt.axis([-2, 15, -2, 15])
		plt.grid(True)
		#plt.pause(self.animationSleep)
		
class Graph(RIG):
	def __init__(self,delta,B,WS,FS,Env,R,agent=None):
		super().__init__(delta,B,WS,FS,Env,R,agent)
		self.SameNodeDistance=0
		self.AllowSelfLoops=False
		self.limitDensity=False
	
	def updatePathList(self,n_prev,n_new,E,new_edge_list):
		#psuedo cod for paper
		#if pathListEmpty(n_new)
		#	for edge in new_edge_list
		#		pathList(n_new) \gets pathList(n_new) uninon edgeToPathList(edge)
		#else
		#	for path in pathList(n_prev)
		#		extPaths \gets extendPath(path,new_edge_list)
		#		extPaths \gets Prune(extPaths)
		#		pathList(n_new) \gets pathList(n_new) uninon extPaths
		edgeID=(n_prev.idx,n_new.idx)
		#print("updating path",edgeID)
		
		_,highestInfo,_,_=self.bestPath
		if len(n_new.pathList)==0 and edgeID[0]==0:#create path list
			for edge in new_edge_list:
				sn,en,info,edgeBudget,time,uncertainty,primList=edge
				if edgeBudget>self.B:
					continue
				if edgeID in E:
					E[edgeID].append(edge)
				else:
					E[edgeID]=[edge]
				edge_idx=E[edgeID].index(edge)
				if not self.ModularCost:
					self.V[n_new.idx]=n_new
					if len(self.V)>1 and self.bestPath[-1]==None:
						info=self.agent.CalcCost(self.V,self.E,[(edgeID[0],edgeID[1],edge_idx,time,edgeBudget,0)])
					else:
						info=-10000
				n_new.pathList.append([(edgeID[0],edgeID[1],edge_idx,time,edgeBudget,info)])#start node,end node,edge index
				n_new.minBudgetPath=[(edgeID[0],edgeID[1],edge_idx,time,edgeBudget,info)]
				n_new.maxBudgetPath=[(edgeID[0],edgeID[1],edge_idx,time,edgeBudget,info)]
				if highestInfo==None:
					highestInfo=n_new.info=info
					self.bestPath=(edgeBudget,info,n_new.idx,len(n_new.pathList)-1)
				elif info>highestInfo or (info==highestInfo and self.bestPath[0]>edgeBudget):
					highestInfo=n_new.info=info
					self.bestPath=(edgeBudget,info,n_new.idx,len(n_new.pathList)-1)
			#if edgeID[0]!=0:
			#	print("looky looky!",new_edge_list)
			#	print(n_prev)
			#	print(n_new)
			#	input()
		else:#update path list
			tempPathlist=[]
			if len(n_new.pathList)==len(n_prev.pathList) and len(n_new.pathList)>0:
				if set(n_new.pathList[0])==set(n_prev.pathList[0]) and set(n_new.pathList[-1])==set(n_prev.pathList[-1]):
					comboList=n_new.pathList
				else:
					comboList=n_new.pathList+n_prev.pathList
			else:
				comboList=n_new.pathList+n_prev.pathList
			for p in comboList:
				#print(p)
				#if p[0][0]!=0:
				#	print(edgeID,p)
				#	print(n_new,n_prev)
				#	print((n_new.pathList))
				#	print(n_prev.pathList)
				#	input()
				if p[-1][1]==edgeID[0]:
					for edge in new_edge_list:
						if edgeID in E:
							E[edgeID].append(edge)
						elif edgeID[1] in self.Vidx:
							E[edgeID]=[edge]
						sn,en,info,edgeBudget,time,uncertainty,primList=edge
						if self.ModularCost:
							pathInfo=p[-1][5]+info
						pathTime=p[-1][3]+time
						if p[-1][4]<0:
							pathBudget=edgeBudget
						else:
							pathBudget=p[-1][4]+edgeBudget
						if pathBudget<n_new.minPathCost or np.isinf(n_new.minPathCost):
							n_new.minPathCost=pathBudget						
						if pathBudget<self.B:
							if edgeID in E and not (edgeID[1] in self.Vidx):
								E[edgeID].append(edge)
							else:
								E[edgeID]=[edge]
							edge_idx=E[edgeID].index(edge)
							if not self.ModularCost:
								if not n_new.idx in self.Vidx:
									self.V[n_new.idx]=n_new
								if len(self.V)>1 and pathBudget>self.budgetCutoff*self.B:
									pathInfo=self.agent.CalcCost(self.V,self.E,p.copy()+[(edgeID[0],edgeID[1],edge_idx,pathTime,pathBudget,0)])
								else:
									pathInfo=-10000
							#if pathBudget<=self.bestPath[0] or pathInfo>=highestInfo:
							#	tempPathlist.append(p.copy()+[(edgeID[0],edgeID[1],edge_idx,pathTime,pathBudget,pathInfo)])
							if pathInfo>highestInfo or (pathInfo==highestInfo and self.bestPath[0]>pathBudget):
								highestInfo=n_new.info=pathInfo
								self.bestPath=(pathBudget,pathInfo,n_new.idx,len(tempPathlist))
							#if pathBudget<=self.bestPath[0] or pathInfo>=self.bestPath[1]:
							tempPathlist.append(p.copy()+[(edgeID[0],edgeID[1],edge_idx,pathTime,pathBudget,pathInfo)])
						#print("edge",edge)
				elif p[0][0]==0:
					tempPathlist.append(p)
			#if len(tempPathlist)==0 and n_new.idx in self.Vidx:
			#	self.Vidx.remove(n_new.idx)
			#	print(edgeID)
			#	print(new_edge_list)
			#	print("temp: ",tempPathlist)
			#	print("n_new: ",n_new,len(n_new.pathList))
			#	print("n_prev: ",n_prev,len(n_prev.pathList))
			#	input()
			n_new.pathList=tempPathlist
			
	def plan(self,xstart,R=None,Rd=0):#plan from scratch
		#Initialize cost, information, starting node, node list, edge list, and graph
		if R==None:
			R=self.R
		delta,B=(self.delta,self.B)
		if self.animate:
			self.draw_graph(self.V,self.E,self.WS)
		n=Node(xstart)
		n.path.append(n.idx)
		self.V=V={n.idx:n}
		Vidx={n.idx}#index set for nodes in V
		Vc=self.Vc
		E=self.E
		self.agent.logDetPrior = None
		while self.terminalCond():
			tempShow=self.curIter<20 or self.curIter%10
			if self.debugMode:
				print("iteration ",self.curIter)
				print("Sample configuration space of vehicle")  
			xsamp=self.Sample(self.WS)
			#xsamp=np.array([[4],[4]])
			if self.debugMode:
				print("random sample: ",xsamp)
			if self.animate and tempShow:
				self.draw_graph(self.V,self.E.keys(),self.WS,xsamp)
				self.drawBestPath()
				plt.pause(self.animationSleep)
			if self.debugMode:
				print("find  node nearest to the sampled point")
			n_nearest=self.Nearest(xsamp,Vidx.difference(Vc),Rd)
			if self.debugMode:
				print("x_nearest: ",n_nearest)
			x_nearest=n_nearest.state
			#print("drive robot towards sample point")
			xfeas=self.Steer(x_nearest,xsamp,delta)
			if self.debugMode:
				print("xfeas: ",xfeas)
			t_nearIdx,Nnear=self.Near(xfeas,Vidx.difference(Vc),R,withNearest=True)
			if t_nearIdx>-1:
				if self.agent.getStateDistance(V[t_nearIdx].state,xfeas)<self.SameNodeDistance:
					xfeas=V[t_nearIdx].state
			if self.NoCollision(x_nearest,xfeas,self.FS):
				prune,n_new,e_new=self.createNode(n_nearest,xfeas,Vidx)
				if self.agent.getStateDistance(x_nearest,xfeas)<self.SameNodeDistance:
					if self.debugMode:
						print("to close to node {0}.".format(n_nearest.idx))
					n_new=n_nearest
					xfeas=x_nearest
				elif t_nearIdx>-1:
					if self.agent.getStateDistance(V[t_nearIdx].state,xfeas)<self.SameNodeDistance:
						if self.debugMode:
							print("to close to node {0}.".format(n_nearest.idx))
						n_new=V[t_nearIdx]
						xfeas=V[t_nearIdx].state
				if self.debugMode:
					print("no collision! Calculate new information, cost, and create node",n_new.idx)
				new_edge_list=self.agent.edgePlanner(n_nearest,n_new,self.Env)
				if len(new_edge_list)>0:
					#print("new edge list: ",new_edge_list)
					#print("pre update",n_new,len(n_new.pathList))
					self.updatePathList(n_nearest,n_new,E,new_edge_list)
					#print("post update",n_new,len(n_new.pathList))
					if len(n_new.pathList)>0:
						#if n_new.minPathCost<0:
						#	print(n_new)
						#	print(n_new.pathList)
						#	print(new_edge_list)
						#	input('w')
						V[n_new.idx]=n_new
						Vidx.add(n_new.idx)
						self.V=V
						self.E=E
					#add to closed list if budget exceeded
					if n_new.minPathCost>B:# np.sum(n_new.cost>B):#doing this in case Budget and cost are vector valued
						if self.debugMode:
							print("budget exceeded! Don't extend this trajectory further")
						pass#Vc.add(n_new.idx)
			else:
				if self.debugMode:
					print("collision! find a new path.")
				continue
			if self.animate and tempShow:
				self.draw_graph(self.V,self.E.keys(),self.WS,xsamp)
				self.drawBestPath()
				plt.pause(self.animationSleep)
			#print("Find points near new nodes to be extended")
			if self.debugMode:
				print("Check if these nodes can be extended to new nodes. Nodes to check: ",len(Nnear))
			#Nnear=self.Near(xfeas,Vidx.difference(Vc),R)
			for n_near in Nnear:
				if n_near.idx==n_new.idx:
					if self.debugMode:
						print("self loop")
					if not self.AllowSelfLoops:
						continue
				if self.debugMode:
					print("Extending {0} towards {1}".format(n_near.idx,n_new.idx))
				xnear=n_near.state
				xnew=self.Steer(xnear,self.agent.getpstate(xfeas),delta)
				if self.debugMode:
					print(xnew)
					print(n_near)
					print(n_new)
				if self.NoCollision(xnear,xnew,self.FS):
					#Calculate new information and cost
					prune,n_new2,e_new=self.createNode(n_near,xnew,Vidx)
					if self.agent.getStateDistance(xfeas,xnew)<self.SameNodeDistance:
						if self.debugMode:
							print("to close to node {0}.".format(n_nearest.idx))
						n_new2=n_new
						xnew=xfeas
					if prune and self.debugMode:
						print("pruning new2 node:" ,n_new2.idx)
					if not prune:
						if self.debugMode:
							print("no collision! Calculate new information, cost, and create node",n_new2.idx)
						
						new_edge_list=self.agent.edgePlanner(n_near,n_new2,self.Env)
						if len(new_edge_list)>0:
							#edgeID=(n_near.idx,n_new2.idx)
							#if edgeID in E:
							#	E[edgeID]=E[edgeID]+new_edge_list
							#else:
							#	E[edgeID]=new_edge_list
							#print("new edge list: ",new_edge_list)
							#print("near loop: pre update",n_new2,len(n_new2.pathList))
							self.updatePathList(n_near,n_new2,E,new_edge_list)	
							#print("near loop: post update",n_new2,len(n_new2.pathList))
							if len(n_new2.pathList)>0:
								#if n_new2.minPathCost<0:
								#	print(n_new2)
								#	print(n_new2.pathList)
								#	print(new_edge_list)
								#	input('near loop')
								V[n_new2.idx]=n_new2
								Vidx.add(n_new2.idx)
								self.V=V
								self.E=E
						#add to closed list if budget exceeded
						if n_new2.minPathCost>B:# np.sum(n_new2.cost>B):#doing this incase Budget and cost are vector valued
							if self.debugMode:
								print("budget exceeded! Don't extend this trajectory further")
							pass#Vc.add(n_new2.idx)
				if self.animateNewEdge and self.animate and tempShow:
					self.draw_graph(self.V,self.E.keys(),self.WS)
					self.drawBestPath()
					plt.pause(self.animationSleep)
		#print(V)
		#print(self.node_locs())
		self.V=V
		self.E=E
		self.Vc=Vc
		self.Vidx=Vidx
		#print("num edges",len(E))
		#print("edge List",E)
		#print("edge List:",E.keys())
		#for k in E.keys():
		#	if len(E[k])>2:
		#		print("multiedge:",k,len(E[k]))
		#print("Closed Nodes",Vc)
		#print("num nodes",len(Vidx))
		#print("path for ",max(Vidx),": ",V[max(Vidx)].path)
		#print("path list for ",max(Vidx),": ",len(V[max(Vidx)].pathList))
		#print(V[max(Vidx)])
		
	def cplan():#continue planing
		pass
		
	def childlessNodes(self):
		EdgeArray=np.array(list(self.E.keys()))
		#print(EdgeArray[np.argsort(EdgeArray[:,0])])
		#print(EdgeArray[np.invert(np.in1d(EdgeArray[:,1],EdgeArray[:,0])),1])
		return EdgeArray[np.invert(np.in1d(EdgeArray[:,1],EdgeArray[:,0])),1]
		
	def DFS(self):
		startNode=self.V[0]
		currentNode=self.V[0]
		EdgeArray=np.array(list(self.E.keys()))
		tempEdgeArray=EdgeArray.copy()
		chain=[currentNode.idx]
		visited=np.zeros(len(self.V))
		print("cNode:",currentNode.idx)
		#while currentNode.idx in EdgeArray[:,0]:#traverse 
		tempPath=[]
		while len(chain):#traverse 
			currentNode=self.V[chain[0]]
			tempPath.append(currentNode.idx)
			chain.remove(chain[0])
			children=tempEdgeArray[currentNode.idx==tempEdgeArray[:,0],1]
			if not visited[currentNode.idx]:
				visited[currentNode.idx]=1
			for k in children:
				if not visited[k]:
					chain.insert(0,k)
		return nodes
	
	def BFS(self):
		startNode=self.V[0]
		currentNode=self.V[0]
		EdgeArray=np.array(list(self.E.keys()))
		tempEdgeArray=EdgeArray.copy()
		chain=[currentNode.idx]
		visited=np.zeros(len(self.V))
		print("cNode:",currentNode.idx)
		#while currentNode.idx in EdgeArray[:,0]:#traverse 
		nodes=[]
		while len(chain):#traverse 
			currentNode=self.V[chain[0]]
			nodes.append(currentNode.idx)
			chain.pop()
			children=tempEdgeArray[currentNode.idx==tempEdgeArray[:,0],1]
			if not visited[currentNode.idx]:
				visited[currentNode.idx]=1
			for k in children:
				if not visited[k]:
					chain.append(k)
		return nodes			
			
		
	def search(self):
		startNode=self.V[0]
		currentNode=self.V[0]
		EdgeArray=np.array(list(self.E.keys()))
		tempEdgeArray=EdgeArray.copy()
		chain=[currentNode.idx]
		visited=np.zeros(len(self.V))
		vchild={}
		vparent={}
		#while currentNode.idx in EdgeArray[:,0]:#traverse 
		tempPath=[]
		while len(chain):#traverse 
			currentNode=self.V[chain[-1]]
			tempPath.append(currentNode.idx)
			print("cNode:",currentNode.idx)
			#chain.pop()
			children=tempEdgeArray[currentNode.idx==tempEdgeArray[:,0],1]
			if not currentNode.idx in vchild.keys():
				vchild[currentNode.idx]= [children,np.zeros_like(children)]		
			print("children:",vchild[currentNode.idx])
			
			if not(currentNode.idx in EdgeArray[:,0]):
				print("ec:",chain)
				
				
			if not visited[currentNode.idx]:
				visited[currentNode.idx]=1
			
			candidateNodes=vchild[currentNode.idx][0][vchild[currentNode.idx][1]==0]
			if candidateNodes.shape[0]==0:
				chain.pop()
			else:
				chain.append(candidateNodes[0])
				vchild[currentNode.idx][1][vchild[currentNode.idx][0]==chain[-1]]=1
				

if __name__ == "__main__":
	def WRBFField(x,p,L,s,w):#weighted radial point source
		#w=[sx,sy,sz];
	#     d=s*vecnorm([x,z*ones(size(x,1),1)].*zscale-p.*zscale,2,2);
		d=s*np.linalg.norm((x-p)*w,2,axis=1)
	#     y=sum(min(L,L./d.^2))
		return sum(L*np.exp(-d**2))


	def vectorWRBFField(x,p,L,s,w):
		y=np.zeros((len(x),1))
		for i in range(len(x)):
			y[i,0]=WRBFField(x[i,:],p,L,s,w);
		return y
	def gaussSumFieldSVector(x,p,L):
		y=np.zeros((len(x),1))
		for i in range(len(x)): 
			y[i,0]=gaussSumFieldSingle(x[i,:],p,L)
		return y

	def gaussSumFieldSingle(x,p,L):
		#print(x)
		d=5*np.linalg.norm(x-p,2,axis=1)
		#y=sum(min(L,L/d**2))
		y=np.sum(L*np.exp(-d**2))
		return y
	def dxdt(t,zx,u1,u2):#dynamics
		x,y,yaw,E=zx
		yaw=angleWrap(yaw,np.pi)
		v,r=(u1,u2)
		dx=v*np.cos(yaw)
		dy=v*np.sin(yaw)
		dyaw=r
		dE=0.5*v**2+0.5*(r**2/5)
		return [dx,dy,dyaw,dE]
	
	
	
	
	print("planning")
	WS=np.array([[0,3],[0,4.5]])
	FS=WS
	p_=np.array([[2,3.25,1],[1,1.15,1]])
	#Env=lambda x:gaussSumFieldSingle(x,p_,25)
	
	L,s,w=(25,5,.5*np.array([3,3,1]))
	Env=lambda x: WRBFField(x,p_,L,s,w)
	specs=[[WS[0,0],WS[0,1],25],[WS[1,0],WS[1,1],50]]
	dim = len(specs)
	#print(specs,dim)
	grid = np.meshgrid(*[np.linspace(specs[i][0], specs[i][1], specs[i][2]) for i in range(dim)])
	ss = np.array([grid[i].ravel() for i in range(dim)]).T
	p=gaussSumFieldSVector(ss,np.array([[2,3.25],[1,1.15]]),25)
	
	specs3D=[[WS[0,0],WS[0,1],10],[WS[1,0],WS[1,1],20],[0,1,5]]
	dim = len(specs3D)
	grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
	ss3D = np.array([grid3D[i].ravel() for i in range(dim)]).T
	#EID=vectorWRBFField(ss3D,p_,L,s,w)
	EID=ekld.softmax(vectorWRBFField(ss3D,p_,L,s,w))
	#print(EID)
	
	grace=GraceAgent()
	grace.CalcCost=grace.calcPathErgodicity
	#grace.CalcCost=grace.self.calcPathInfo 
		
	#legTypes=["Spiral","Glide","Swim","FlatDive"]
	grace.legProbs=[0,1/3,1/3,1/3]
	#grace.legProbs=[0,1/10,8/10,1/10]
	grace.legProbs=[1/3,1/3,1/3,0]
	grace.trajCount=3
	grace.measRate=.1
	grace.EID=EID
	grace.ergSigma=0.1*np.eye(3)
	grace.fieldGrid=ss3D
	grace.underWaterTimeLimit=60*3
	B=100
	nearRad=.25
	
	stepSize=2
	xstart=np.array([[0],[0]])
	#agent=Geometric2DAgent()
	agent=grace
	planner=Graph(stepSize,B,WS,FS,Env,nearRad,agent=agent)
	planner.AllowSelfLoops=False
	planner.animate = True
	planner.animateNewEdge = False
	planner.debugMode = False		
	planner.animationSleep=0.01
	planner.maxIter=50
	planner.SameNodeDistance=.1
	planner.EvaluateCost=agent.getCost
	
	ts=time.time()
	planner.plan(xstart,Rd=1)
	print("graph building time: ",time.time()-ts)
	pathBudget,pathInfo,bnode_idx,bpath_idx=planner.bestPath
	planner.draw_graph(planner.V,planner.E,planner.WS)
	p.shape=grid[0].shape
	plt.gcf()
	plt.contour(grid[0],grid[1],p)
	#planner.draw_path(planner.V,planner.V[max(planner.Vidx)].path)
	#planner.draw_path(planner.V,planner.V[bnode_idx].path)
	planner.draw_2D_path_projection(planner.V,planner.V[bnode_idx].pathList[bpath_idx])
	#print(planner.V[max(planner.Vidx)].pathList[0])
	#planner.draw_3D_path(planner.V,planner.E,planner.V[max(planner.Vidx)].pathList[0],dense=True)
	#planner.draw_3D_path(planner.V,planner.E,planner.V[max(planner.Vidx)].pathList[0],dense=False)
	print("most informative path",planner.bestPath)
	planner.draw_3D_path(planner.V,planner.E,planner.V[bnode_idx].pathList[bpath_idx],dense=True)
	planner.draw_3D_path(planner.V,planner.E,planner.V[bnode_idx].pathList[bpath_idx],dense=False)
	print("number of paths for node ",bnode_idx,": ",len(planner.V[bnode_idx].pathList))
	print(planner.V[bnode_idx].pathList[bpath_idx])
	#print(planner.V[bnode_idx].pathList[bpath_idx+1])
	print(planner.V[bnode_idx].pathList[bpath_idx-1])
	print(planner.V[bnode_idx])
	input()
