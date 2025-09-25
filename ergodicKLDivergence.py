import numpy as np
import scipy
import scipy.integrate
import scipy.stats

def softmax(a):
	''' converts a vector to a probability distribution'''
	ea=np.exp(a)
	return ea/np.sum(ea)


def config_ss(*specs):
	"""Specifying search space.
	specs -- Specification for each dimension, in format of: (low, high, num)
	Example usage:
		2d space -> ss,x,y,Lx,Ly=config([0,1,10],[0,1,10]) 
		3d space -> ss,x,y,z,Lx,Ly,Lz=config([0,1,10],[0,1,10],[0,1,2])
	"""
	dim = len(specs)
	#print(specs,dim)
	_grid = np.meshgrid(*[
		np.linspace(specs[i][0], specs[i][1], specs[i][2])
		for i in range(dim)
	])
	L=[specs[i][1]- specs[i][0]for i in range(dim)]
	_ss = np.array([
		_grid[i].ravel() for i in range(dim)
	]).T

	_ret = _ss, *_grid,*L
	return _ret
	

def gaussianSensor(x,s,Sigma):
	#x - matrix where each row is a point on the trajectory
	#s - row vector representing a single point in the domain
	#Sigma - covariance assumed to be a diagonal matrix
	#	   - alternertively, sigma can be a matrix with each row coresponding to the diagonal elements of covariance for each point
	d=s.shape[0]
	if Sigma.shape[0]==Sigma.shape[1]:
		return (1/np.sqrt((2*np.pi)**d*np.linalg.det(Sigma)))*np.exp(-0.5*np.sum((x-s)**2/np.diag(Sigma),1))
		#return (1/np.sqrt((2*np.pi)**d*np.linalg.det(Sigma)))*np.exp(-np.linalg.norm((s-x)/np.sqrt(np.diag(Sigma)),2,1)**2)
	elif Sigma.shape[0]==x.shape[0]:
		return (1/np.sqrt((2*np.pi)**d*np.prod(Sigma,axis=1)))*np.exp(-0.5*np.sum((x-s)**2/Sigma,1),2,1)

def computeTrajectoryIntegrand(t,x,s,Sigma):
	#computes time averaged statistics of a trajectory x and on a discrete doiman s   
	#t - vector of time stamps for each point on the trajectory
	#x - matrix where each row is a point on the trajectory
	#s - matrix with each row representing a single point in the domain
	#Sigma - covariance assumed to be a diagonal matrix
	#	   - alternertively, sigma can be a matrix with each row coresponding to the diagonal elements of covariance for each point
	p=np.zeros((s.shape[0],1))
	for i in range(s.shape[0]-1):
		#print(t,x,s[i:i+1,:])
		#print(x.shape,s[i:i+1,:].shape)
		#print(t[:,0],gaussianSensor(x,s[i:i+1,:],Sigma))
		#print(gaussianSensor(x[0:1,:],s[i:i+1,:],Sigma))
		#print(np.trapz(t[:,0],gaussianSensor(x,s[i:i+1,:],Sigma)))
		p[i]=np.trapz(gaussianSensor(x,s[i:i+1,:],Sigma),t[:,0])
	return p/(t[-1,0]-t[0,0])

def ergodicDivergence(p,q):
	#calculates kl divergence of two discrete probability distributions p and q
	#normalizes
	#if p is trajectory distribution and q is target distribution,calculates backward kl divergence (mode seeking behavior)
	#if q is trajectory distribution and p is target distribution,calculates forward kl divergence (mean seeking behavior)
	return scipy.stats.entropy(p,q)[0]

def computeCombinedTrajDist(dur1,dur2,q1,q2):
	return dur1/(dur1+dur2)*q1+dur2/(dur1+dur2)*q2
if __name__=="__main__":
	import matplotlib.pyplot as plt
	def field2(x,p,L):
		y=np.zeros((len(x),1))
		for i in range(len(x)): 
			y[i,0]=field1(x[i,:],p,L)
		return y

	def field1(x,p,L):
		d=5*np.linalg.norm(x-p,2,axis=1)
		#y=sum(min(L,L/d**2))
		y=np.sum(L*np.exp(-d**2))
		return y
	numSources=5
	sourceDim=2
	sourceLocs=np.array([[0.305233157495829,	1.07699174082087],[3.11967516896046,	1.00224092731988],[1.75363692576357,	0.144102266719523],[2.89386071132377,	0.536877960203742],[3.91195804798641,	0.999765001651120]])
	samplePoints=np.array([[0,0],[0,1],[2,1],[2,2],[3,1]])
	t=np.array([[0],[1],[2],[3],[4]])
	gridNums=[50, 50]
	ss,gridx,gridy,lx,ly=config_ss([0,4,gridNums[0]],[0,2,gridNums[1]])
	print(ss.shape)
	p=1e-15+field2(ss,sourceLocs,25)
	q=1e-15+computeTrajectoryIntegrand(t,samplePoints,ss,np.diag([.01,.01]))
	#print(np.concatenate((ss,p,q),axis=1))
	print(max(p),min(p),max(q),min(q),ergodicDivergence(p,q),ergodicDivergence(q,p),p.shape,q.shape)
	plt.clf()
	p.shape=gridx.shape
	plt.contour(gridx,gridy,p)
	plt.plot(samplePoints[:,0],samplePoints[:,1],"og")
	plt.figure()
	q.shape=gridx.shape
	plt.contour(gridx,gridy,q)
	plt.show()
	input()