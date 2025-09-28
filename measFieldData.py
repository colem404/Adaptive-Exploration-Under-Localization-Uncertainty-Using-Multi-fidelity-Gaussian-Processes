# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 19:01:24 2025

@author: dac00
"""
# code takes a .csv file with a trajectory (t,x,y,z) and produces measuremeants from a simulated measurement field.
import numpy as np
import exploreSimSettings as ess
import glob, os


file_list=os.listdir(os.getcwd()+"/Data/TrajectoriesAndEstimates/")
savePath="Data/TrajectoriesAndEstimates/FieldData/"
for file_name in file_list:
    if file_name.endswith(".csv"):
        print(file_name)

        file_path=os.getcwd()+"/Data/TrajectoriesAndEstimates/"+file_name
        with open(file_path, 'r') as f:
                headers = f.readline().strip().split(',')
                print(headers)
                trajData = np.loadtxt(f, delimiter=',')
        	
        #setup measurment field
        xmax=max(10,np.max(trajData[:,[headers.index("x")]]))
        ymax=max(20,np.max(trajData[:,[headers.index("y")]]))
        WS=np.array([[0,xmax],[0,ymax]])
        maxDepth=max(10,np.max(trajData[:,[headers.index("z")]]))
        p=np.array([[np.random.rand()*WS[0,1],np.random.rand()*WS[1,1],np.random.rand()*maxDepth],[np.random.rand()*WS[0,1],np.random.rand()*WS[1,1],maxDepth],[np.random.rand()*WS[0,1],np.random.rand()*WS[1,1],np.random.rand()*maxDepth],[np.random.rand()*WS[0,1],np.random.rand()*WS[1,1],.3*maxDepth],[np.random.rand()*WS[0,1],np.random.rand()*WS[1,1],np.random.rand()*maxDepth]])
        L,s,w=(10*np.random.rand(),0.5*np.random.rand(),0.5*np.array([5*np.random.rand(),5*np.random.rand(),5*np.random.rand()]))
        measFunc=lambda x: ess.WRBFField(x,p,L,s,w)
        FS=WS
        
        f=open(savePath+"FieldSettings"+str(ess.seed)+".txt","w")
        f.write("Type: WRBFField\n")
        f.write("WS: "+str(WS)+"\n")
        f.write("maxDepth: "+str(maxDepth)+"\n")
        f.write("L,s,w: "+str((L,s,w))+"\n")
        f.write("sources:\n"+str(p)+"\n")
        f.write("measNois:"+ str(ess.measNois)+"\n")
        f.close()
        	
        
        													  
        xnow=trajData[0,[headers.index("x"),headers.index("y"),headers.index("z")]]
        u=[0,0,0]#vx,vy,vz
        t=0
        #setup kalman filter for position 
        xhat=np.array([xnow.flatten().tolist()[0:3]+[0,0,0]]).T	#x,y,z,vx,vy,vz
        Pxhat=ess.Pxhat2
        Qxhat=ess.Qxhat2
        Axhat=ess.Axhat2
        Bxhat=ess.Bxhat2
        Rxhat=ess.Rxhat2
        #set up dataset
        
        dataFile=open(savePath+"fieldMeas_"+str(ess.seed)+"_"+file_name,'w')
        dataFile.write("t,x,y,z,fieldVal\n")
        
        cnt=trajData.shape[0]
        lastSampleTime=trajData[0,headers.index("t")]
        for j in range(1, cnt):	
        	xtemp=trajData[j-1,[headers.index("x"),headers.index("y"),headers.index("z")]].tolist()
        	t=trajData[j-1,headers.index("t")]
        	ym=max(0,measFunc(np.array([xtemp[0:3]]))+ess.measNois*np.random.normal())
        	tempDataArr=np.array([[t]+xtemp+[ym]])
        	np.savetxt(dataFile,tempDataArr,delimiter=",")
    	
    
    		
    
    
        dataFile.close()

