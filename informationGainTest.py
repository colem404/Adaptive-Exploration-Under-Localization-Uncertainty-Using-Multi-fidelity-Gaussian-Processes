import GPy
import numpy as np

#set random seed for repeatability
#np.random.seed(0)

#test set of points representing the entire function on desired domain
Xpred=np.array([np.arange(-3,3,.1)]).T 
#number of measurements in GP
measNum=3
#measNum=Xpred.shape[0]
#locations of measurements
X = np.random.uniform(-3.,3.,(measNum,1)) 
#X=np.ones((measNum,1))
#X=Xpred
print(X.shape)
print(Xpred.shape)
#calculate noisy measurements
Y = np.sin(X) + np.random.randn(measNum,1)*0.05
priorX=np.array([[-100]])
priorY=np.array([[0]])
kernel = GPy.kern.RBF(input_dim=1, variance=0.7407926234918235, lengthscale=1.5704374366230516)
m = GPy.models.GPRegression(X,Y,kernel)
m.Gaussian_noise.variance=0.0010413149736387451
#print(m)
#m.optimize()
#print(m)
m.set_XY(priorX,priorY)
_,Kprior=m.predict(Xpred,full_cov=1)
_,Kprior2=m.predict(X,full_cov=1)
logDetPrior=np.log(np.linalg.det(Kprior))
logDetPrior2=np.log(np.linalg.det(Kprior2))
m.set_XY(X,Y)
_,Kposterior=m.predict(Xpred,full_cov=1)
_,Kposterior2=m.predict(X,full_cov=1)
I=0.5*(logDetPrior-np.log(np.linalg.det(Kposterior)))#I(f(Xpred);ftrue)
I3=0.5*(logDetPrior2-np.log(np.linalg.det(Kposterior2)))#I(f(X);ftrue)

sig_n=m.Gaussian_noise.variance[0]
xtemp=X[0,:]
xtemp.shape=(1,X.shape[1])
m.set_XY(xtemp,np.array([[0]]))
_,sig_y=m.predict(xtemp)
I2=0.5*np.log(1+sig_y[0,0]/sig_n)#
for i in range(2,X.shape[0]):		
	xtemp=X[i,:]
	xtemp.shape=(1,X.shape[1])
	_,sig_y=m.predict(xtemp)
	I2+=0.5*np.log(1+sig_y[0,0]/sig_n)
	m.set_XY(np.concatenate((m.X,xtemp)),np.concatenate((m.Y,np.array([[0]]))))
	print(m.X.shape,I2)
print(I,I2,I3)