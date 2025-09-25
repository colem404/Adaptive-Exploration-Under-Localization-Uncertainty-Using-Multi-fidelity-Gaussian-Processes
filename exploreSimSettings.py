import numpy as np
import ergodicKLDivergence as ekld
import GraceRIGV3 as IG
import re

def getEID(gp,WS,mD,testSet=None,emu=False,alpha=1.0/11):	
	if testSet==None:
		specs3D=[[WS[0,0],WS[0,1],10],[WS[1,0],WS[1,1],20],[0,mD,10]]
		dim = len(specs3D)
		grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
		ss3D = np.array([grid3D[i].ravel() for i in range(dim)]).T
	else:
		ss3D=testSet
	if emu:
		mu,sig=gp.predict(np.hstack((ss3D,np.ones((ss3D.shape[0],1))*2)))
		prior_sig=np.sum(gp.gpy_model.param_array[[0,4,8,-1]]) #variance with no data
	else:
		mu,sig=gp.predict(ss3D)
		prior_sig=gp.kern.variance[0]+gp.Gaussian_noise.variance[0] #variance with no data
	if auto:
		alpha=1-np.mean(sig)/prior_sig
		#alpha=np.exp(-3*np.mean(sig)/prior_sig)
		#alpha=np.sqrt(prior_sig-np.mean(sig))/np.sqrt(prior_sig)
		#alpha=1-(0.75*np.mean(sig)+0.25*np.max(sig))/prior_sig
	
	#fauxUCB=alpha*mu+(1-alpha)*np.sqrt(sig)
	fauxUCB=alpha*mu+(1-alpha)*np.sqrt(np.abs(sig))
	#fauxUCB=alpha*mu+(1-alpha)*sig
	EID=ekld.softmax(fauxUCB)
	if np.any(sig<0):
		#print(np.hstack((ss3D,np.ones((ss3D.shape[0],1))*2)))
		print("mu:",np.min(mu),np.max(mu),np.mean(mu))
		print("var:",np.min(sig),np.max(sig),np.mean(sig),np.sum(sig<0)/sig.shape[0])
		print("EID:",fauxUCB,np.min(fauxUCB),np.max(fauxUCB),np.mean(fauxUCB))
		EID=EID*0+1/EID.shape[0]
	#print("alphas:",1-np.mean(sig)/prior_sig,np.exp(-5*np.mean(sig)/prior_sig),np.sqrt(prior_sig-np.mean(sig))/np.sqrt(prior_sig),1-(0.75*np.mean(sig)+0.25*np.max(sig))/prior_sig)
	return EID,ss3D


def parse_field_settings(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # --- Extract L, s, w ---
    lsw_line = next((line for line in lines if line.startswith("L,s,w:")), None)
    if lsw_line is None:
        raise ValueError("Missing L,s,w line")
    lsw_str = lsw_line.split(":", 1)[1].strip()
    L, s, w = eval(lsw_str, {"array": np.array})

    # --- Extract sources matrix ---
    # Locate the 'sources:' line
    sources_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("sources:"):
            sources_start = i + 1
            break
    if sources_start is None:
        raise ValueError("No 'sources:' section found in file.")

    # Collect lines until the next key-value line
    sources_lines = []
    for line in lines[sources_start:]:
        if re.match(r"^\w+:", line):  # Detects next field like 'measNois:'
            break
        sources_lines.append(line.strip().replace("[", "").replace("]", ""))

    # Parse as NumPy array
    sources_text = "\n".join(sources_lines)
    sources_matrix = np.loadtxt(sources_text.splitlines())

    return L, s, w, sources_matrix
	
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

seed=5
auto=0
np.random.seed(seed)
fieldOffset=0
#function to interpolate (linear) 3D path
trajPnt= lambda x,y: np.array([np.interp(x,y[:,3],y[:,0]),np.interp(x,y[:,3],y[:,1]),np.interp(x,y[:,3],y[:,2])])
#field settings
#gpHyps=(2.0527883,  1.23383256, 1.97989437, 3.96970347, 0.01541058)#signal variance, lx,ly,lz, noise variance	
gpHyps=(1, 1, 1, 1, 1)#signal variance, lx,ly,lz, noise variance	
WS=np.array([[0,10],[0,20]])
maxDepth=10
FS=WS
p=np.array([[.7*WS[0,1],.7*WS[1,1],.5*maxDepth],[.3*WS[0,1],.2*WS[1,1],maxDepth],[.1*WS[0,1],.9*WS[1,1],maxDepth],[.6*WS[0,1],.1*WS[1,1],.3*maxDepth],[.1*WS[0,1],.1*WS[1,1],maxDepth]])
L,s,w=(10,0.5,0.5*np.array([3,2,1]))
measFunc=lambda x: WRBFField(x,p,L,s,w)+fieldOffset
f=open("Data/fieldSettings.txt","w")
f.write("Type: WRBFField\n")
f.write("L,s,w: "+str((L,s,w))+"\n")
f.write("sources: "+str(p)+"\n")
f.close()
fidlevels=((min(np.diff(WS))*np.array([0.05,0.15,.25]))**2).tolist()
#print(fidlevels)

#evaluation settings
f=lambda x:vectorWRBFField(x,p,L,s,w)+fieldOffset	



specs3D=[[WS[0,0],WS[0,1],10],[WS[1,0],WS[1,1],20],[0,maxDepth,10]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
testPoints = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T
'''
tx,ty,tz=np.meshgrid(np.linspace(WS[0,0],WS[0,1],20),np.linspace(WS[1,0],WS[1,1],10),np.linspace(0,maxDepth,20))
tx,ty,tz=(tx.ravel('F'),ty.ravel('F'),tz.ravel('F'))
tx.shape=(tx.shape[0],1)
ty.shape=(ty.shape[0],1)
tz.shape=(tz.shape[0],1)
X_star1 = np.concatenate((tx,ty,tz),axis=1)
muv2,sigv2=gp.predict(X_star1)
mu2v2,sig2v2=lin_mf_model.predict(np.concatenate((X_star1,np.zeros((X_star1.shape[0],1))),axis=1))
np.savetxt("Data/resultsv2.csv",np.concatenate((tx,ty,tz,f(X_star1),muv2,sigv2,mu2v2,sig2v2),axis=1),delimiter=",",header=" x,y,z,trueField,gpMean,gpVar,emuMean,emuVar")
'''
#sim setup
x0=np.array([[.5],[.5],[0]])#xp,yp,zp
dt=.1

Pxhat=.001*np.eye(4)
Qxhat=np.diag([0.005,0.005,0.005,0.05])
Axhat=lambda dt:np.eye(4)+np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])*dt
Bxhat=np.eye(4)
Bxhat[3,3]=0
Rxhat=np.diag([.1,.1,.05])


x02=np.array([[.5],[.5],[0],[0],[0],[0]])#xp,yp,zp,vx,vy,vz

vmn=0.1#0#0.1#velocity measurment noise (m/s)^2
Pxhat2=.001*np.eye(6)
Qxhat2=np.diag([0.005,0.005,0.005,0.05,0.05,0.05])
Axhat2=lambda dt:np.eye(6)+np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])*dt
Bxhat2=0
Rxhat2=np.diag([.1,.1,.05,0.25,0.25,0.25])
#Rxhat2=np.diag([.1,.1,.05,0.5,0.5,0.5])
kfMeasNoise=np.array([[0.05],[0.05],[0.02],[vmn],[vmn],[vmn]])
#kfMeasNoise=2*np.array([[0.0],[0.0],[0.0],[.125],[.125],[.125]])

umaxmag=.25
atSurface=0.2
measNois=0.125
waves=np.array([.002,.002,0])

#agent settings 
goalVar=2**2
trajCount=3
measRate=.05
SurfaceBySpiral=False
swimSpeed=.3
spiralSpeed=.075
vertGlideSpeed=.075
flatDiveSpeed=.1
FlatDiveEnergy=0.1
GlideEnergy=0.15
tailEnergyScale=.1
timeEnergy=0.005

agent=IG.GraceAgent()
#legTypes=["Spiral","Glide","Swim","FlatDive"]
agent.legProbs=[0,1/3,1/3,1/3]
#agent.legProbs=[0,0,2/3,1/3]
agent.fidLevs=fidlevels
agent.trajCount=trajCount
agent.measRate=measRate
agent.maxDepth=maxDepth
agent.SurfaceBySpiral=SurfaceBySpiral
agent.swimSpeed=swimSpeed
agent.spiralSpeed=spiralSpeed
agent.vertGlideSpeed=vertGlideSpeed
agent.flatDiveSpeed=flatDiveSpeed
# for budget calculation
agent.FlatDiveEnergy=FlatDiveEnergy
agent.GlideEnergy=GlideEnergy
agent.tailEnergyScale=tailEnergyScale
agent.timeEnergy=timeEnergy
#for model with velocity as input
#agent.varianceRate=Qxhat[0,0]
#for constant velocity model
agent.varianceRate=Qxhat[0,0]+Qxhat2[3,3]**2
agent.underWaterTimeLimit=(goalVar)/agent.varianceRate

#planner settings
B=150
BD=10#BudgetDivisor
SameNodeDistance=1
maxIter=100
Rd=5
nearRad=1.25
stepSize=10
