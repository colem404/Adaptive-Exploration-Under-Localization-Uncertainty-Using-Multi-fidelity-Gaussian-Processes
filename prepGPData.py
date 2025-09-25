# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 19:01:24 2025

@author: dac00
"""
# code takes a .csv file with a trajectory (t,x,y,z) and produces measuremeants from a simulated measurement field.
import numpy as np
import exploreSimSettings as ess
import os
# import GPy
# import emukit.multi_fidelity 
# from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
# from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
# from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper

measRate=.2#hz
basePath="/Data/TrajectoriesAndEstimates/"
savePath="Data/TrajectoriesAndEstimates/GPDataSets/"
fieldPath="/Data/TrajectoriesAndEstimates/FieldData/"
file_list=os.listdir(os.getcwd()+fieldPath)
#glob.glob(os.getcwd()+fieldPath+"*.txt")
for file_name in file_list:
    if file_name.endswith(".csv"):
        print(file_name)
        
        field_file_path=os.getcwd()+fieldPath+file_name
        with open(field_file_path, 'r') as f:
                field_headers = f.readline().strip().split(',')
                print(field_headers)
                fieldData = np.loadtxt(f, delimiter=',')
                
        fc=file_name.split("_")
        file_path=os.getcwd()+basePath+fc[-2]+"_"+fc[-1]
        with open(file_path, 'r') as f:
                headers = f.readline().strip().split(',')
                print(headers)
                trajData = np.loadtxt(f, delimiter=',')
        
        

	

													  
        fidlevels=ess.fidlevels
        #set up dataset
        dataFile=open(savePath+"GPData_"+str(measRate)+"_"+file_name,'w')
        dataFile.write("t,x,y,z,xh,yh,zh,fieldVal,fidLev\n")
        
        cnt=trajData.shape[0]
        lastSampleTime=trajData[0,headers.index("t")]
        for j in range(1, cnt):	
        	xtemp=trajData[j-1,[headers.index("x"),headers.index("y"),headers.index("z")]].tolist()
        	t=trajData[j-1,headers.index("t")]
        	addMeas = t-lastSampleTime>1/measRate
        	if addMeas:
        		lastSampleTime=t
        		Pxhat=np.diag(trajData[j,[headers.index("sigx"),headers.index("sigy")]])
        		covComp=0.5*np.trace(Pxhat)
        		if covComp <fidlevels[0]:
        			fidLev=1
        		elif covComp<fidlevels[1]:
        			fidLev=2
        		else:#elif covComp<fidlevels[2]:
        			fidLev=3
        		xhattemp=trajData[j-1,[headers.index("xh"),headers.index("yh"),headers.index("zh")]].tolist()
        		ym=fieldData[j-1,field_headers.index("fieldVal")]
        		tempDataArr=np.array([[t]+xtemp+xhattemp+[ym,fidLev]])
        		np.savetxt(dataFile,tempDataArr,delimiter=",")
        
        		
        
        
        dataFile.close()

