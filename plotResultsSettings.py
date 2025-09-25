import numpy as np


#field settings	
WS=np.array([[0,10],[0,20]])
maxDepth=10

#evaluation settings
specs3D=[[WS[0,0],WS[0,1],21],[WS[1,0],WS[1,1],21],[0,maxDepth,5]]
dim = len(specs3D)
grid3D = np.meshgrid(*[np.linspace(specs3D[i][0], specs3D[i][1], specs3D[i][2]) for i in range(dim)])
testPoints = np.array([grid3D[i].ravel(('F')) for i in range(dim)]).T



