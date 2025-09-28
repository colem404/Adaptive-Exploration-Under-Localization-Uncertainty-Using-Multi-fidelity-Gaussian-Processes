import numpy as np

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
	
def config_k(*specs):
	"""Specifying Fourier coefficient indices.
	specs -- Specification for each dimension, in format of: (num_k, L), number of coefficients and length of the dimension (L = high - low, from config_ss)
	Example usage:
		2d space ->config([5,1],[5,1])
		3d space ->config([5,1],[5,1],[5,1])
		"""
	dim = len(specs)
	#print(specs,dim)
	_ks = np.meshgrid(*[np.arange(0, specs[i][0], step=1) / specs[i][1] for i in range(dim)])
	_k = np.array([
		_ks[i].ravel() for i in range(dim)
	]).T

	return _k
	
def get_hk(k):
	"""Computing normalization term given coefficient indices.
	k -- Coefficient indices
	"""
	_hk = (2.0*k + np.sin(2.0*k)) / (4.0*k)
	#_hk = np.nan_to_num(_hk, nan=1.0)
	_hk[np.isnan(_hk)]=1.0
	return np.sqrt(np.prod(_hk, axis=1))
	
def get_lamk(k):
	"""Computing Sobolev norm coefficients given Fourier coefficient indices.
	k -- Coefficient indices
	"""
	dim = k.shape[1]
	return (1.0 + np.linalg.norm(k, axis=1)**2) ** (-(dim+1.0) / 2.0)

def _fk(x, k):
	"""(Raw) Fourier basis function (cosine basis). This function will be vectorized in fk().
	x -- Single function input
	k -- Single pair of basis function coefficient
	"""
	return np.prod(np.cos(x * k * np.pi))



def fk(x, k):
	"""Fourier basis function (cosine basis).
	x -- Function inputs, shape should be (N, dim)
	k -- Basis function coefficient pairs, shape should be (M, dim)
	output shape should be (N,M)
	"""
	resd=np.zeros((k.shape[0],x.shape[0]))
	for i in range(k.shape[0]):
		resd[i,:]=np.prod(np.cos(x*k[i,:]*np.pi),1)
	return resd

def get_coefficients(x, w, k, with_hk=False,hk=None):
	"""Computing Fourier coefficients
	x -- A set of points from the search space, e.g. a trajectory. When computing the Fourier coefficients for a function over the search space, it needs to be a discrete grid of the whole search space (e.g. the first returned variable from the "config_ss" function)
	w -- Weights associated with x. If Dirac delta function is used to compute the spatial distribution (time statistics) of trajectories, it is a vector of ones. When computing the Fourier coefficients for a function, it contains function values for each grid cell from x
	k -- Fourier coefficient indices
	"""
	#print(fk(x, k))
	if type(hk)==type(None):
		hk=get_hk(k)
	if with_hk:
		return np.mean(fk(x, k) * w, axis=1) / hk,hk
	return np.mean(fk(x, k) * w, axis=1) / hk

def update_coefficents(coef1,coef2,duration1,duration2):
	"""Combines two sets of Fourier coefficients. Allows calculating coeeficients for an entire trajectory without needing to recalculate coeeficients for the entire history.
	coef1 --- First set of Fourier coefficients
	coef2 --- Second set of Fourier coefficients
	duration1 --- time duration associated with first set of Fourier coefficients
	duration2 --- time duration associated with second set of Fourier coefficients"""
	totTime=duration1+duration2;
	return (duration1*coef1+duration2*coef2)/totTime

def sobolev_norm(coef1, coef2, k):
	"""Computing Sobolev norm between two sets of Fourier coefficients
	coef1 --- First set of Fourier coefficients
	coef2 --- Second set of Fourier coefficients
	k --- Fourier coefficient indices (assumed to same for both sets)
	"""
	return np.sum(get_lamk(k) @ np.square(coef1-coef2))
	
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
	gridNums=[20, 20]
	ss,gridx,gridy,lx,ly=config_ss([0,4,gridNums[0]],[0,2,gridNums[1]])
	print(ss.shape)
	spec2=[[8,lx],[8,ly]]
	K=config_k(*spec2) #this does the same thng as the next line
	K=config_k([8,lx],[8,ly])
	print(K.shape)
	p=field2(ss,sourceLocs,25)
	phik= get_coefficients(ss,p.T,K)
	ck1 = get_coefficients(samplePoints,np.ones((1,samplePoints.shape[0])),K)
	ck2 = get_coefficients(sourceLocs,np.ones((1,samplePoints.shape[0])),K)
	dist_recon = np.matmul(fk(ss,K).T,phik)
	#dist_recon2 = np.matmul(fk(ss,K).T,ck1)
	
	print(sobolev_norm(phik,ck1,K),sobolev_norm(phik,ck2,K))
	plt.clf()
	p.shape=gridx.shape
	plt.contour(gridx,gridy,p)
	plt.plot(samplePoints[:,0],samplePoints[:,1],"og")
	plt.show()
	input()