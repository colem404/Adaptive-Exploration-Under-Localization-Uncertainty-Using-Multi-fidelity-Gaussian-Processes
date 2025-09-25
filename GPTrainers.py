# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 19:17:14 2025

@author: dac00
"""
import numpy as np
import exploreSimSettings as ess
import GPy
import emukit.multi_fidelity 
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
import os
from NIGP import NIGP


basePath="/Data/TrajectoriesAndEstimates/GPDataSets/"
fieldPath="/Data/TrajectoriesAndEstimates/FieldData/"
file_list=os.listdir(os.getcwd()+basePath)
lastSaveFileNum=file_list.index("GPData_0.2_fieldMeas_9_T9_0.csv")
file_list=file_list[lastSaveFileNum:]
savePath="/Data/TrajectoriesAndEstimates/GPResults/"

for file_name in file_list:
    if file_name.endswith(".csv"):
        print(file_name)

        with open(os.getcwd()+basePath+file_name, 'r') as f:
                headers = f.readline().strip().split(',')
                print(headers)
                GPData = np.loadtxt(f, delimiter=',')
                settingsFile="FieldSettings"+file_name.split("_")[3]+".txt"
        
        
        Normalize=1
        GPData=GPData[GPData[:,headers.index("t")]<3600]
        df1=GPData[:,headers.index("fidLev")]==1
        df1=GPData[df1,:]
        df2=GPData[:,headers.index("fidLev")]==2
        df2=GPData[df2,:]
        df3=GPData[:,headers.index("fidLev")]==3
        df3=GPData[df3,:]
        
        Xf1=df1[:,[headers.index("x"),headers.index("y"),headers.index("z")]]
        Xf2=df2[:,[headers.index("x"),headers.index("y"),headers.index("z")]]
        Xf3=df3[:,[headers.index("x"),headers.index("y"),headers.index("z")]]
        Xhf1=df1[:,[headers.index("xh"),headers.index("yh"),headers.index("zh")]]
        Xhf2=df2[:,[headers.index("xh"),headers.index("yh"),headers.index("zh")]]
        Xhf3=df3[:,[headers.index("xh"),headers.index("yh"),headers.index("zh")]]
        y1=df1[:,[headers.index("fieldVal")]]
        y2=df2[:,[headers.index("fieldVal")]]
        y3=df3[:,[headers.index("fieldVal")]]
        
        Xs=[Xf3,Xf2,Xf1]
        Xhs=[Xhf3,Xhf2,Xhf1]
        ys=[y3,y2,y1]
        
        n_fids=3
        X_train, Y_train = convert_xy_lists_to_arrays([Xf3, Xf2, Xf1], [y3, y2, y1])
        Xh_train, Y_train = convert_xy_lists_to_arrays([Xhf3, Xhf2, Xhf1], [y3, y2, y1])
        kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel, n_fidelities=n_fids)
        lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)
        lin_mf_model.set_data(Xh_train,Y_train)
        lin_mf_model.gpy_model.kern.scale.fix([1,1])
        lin_mf_model.optimize()
        emuHypVec=lin_mf_model.gpy_model.param_array.copy()
        print(emuHypVec,emuHypVec.shape)
        f=open(os.getcwd()+savePath+file_name.replace(".csv","_")+"emuGP.txt","w")
        emuHypVec.shape=(1,emuHypVec.shape[0])
        np.savetxt(f,emuHypVec,delimiter=",")
        f.close()
        
        
        sfXhs = GPData[:,[headers.index("xh"),headers.index("yh"),headers.index("zh")]]
        sfXs = GPData[:,[headers.index("x"),headers.index("y"),headers.index("z")]]
        sfys = GPData[:,[headers.index("fieldVal")]]
        kernel=GPy.kern.RBF(input_dim=3,ARD=True)
        gp=GPy.models.GPRegression(sfXhs,sfys,kernel)
        HypNames=gp.parameter_names()
        gp.set_XY(sfXhs,sfys)
        gp.optimize()
        print(gp.param_array)
        f=open(os.getcwd()+savePath+file_name.replace(".csv","_")+"sfGP.txt","w")
        np.savetxt(f,gp.param_array,delimiter=",")
        f.close()
        
        kernelTP=GPy.kern.RBF(input_dim=3,ARD=True)
        gpTruePos=GPy.models.GPRegression(sfXs,sfys,kernelTP)
        HypNames=gpTruePos.parameter_names()
        gpTruePos.set_XY(sfXs,sfys)
        gpTruePos.optimize()
        print(gpTruePos.param_array)
        f=open(os.getcwd()+savePath+file_name.replace(".csv","_")+"sfGPTP.txt","w")
        np.savetxt(f,gpTruePos.param_array,delimiter=",")
        f.close()
        
        nigp = NIGP(n_restarts=2, iters=10, verbose=True)
        nigp.fit(sfXhs, sfys)
        f=open(os.getcwd()+savePath+file_name.replace(".csv","_")+"nisfGP.txt","w")
        np.savetxt(f,nigp.get_params(),delimiter=",")
        f.close()
        
        
        L,s,w,p=ess.parse_field_settings(os.getcwd()+"/Data/TrajectoriesAndEstimates/FieldData/"+settingsFile)
        print("L:", L)
        print("s:", s)
        print("w:", w)
        print("sources:\n", p)
        trueField=lambda x:ess.vectorWRBFField(x,p,L,s,w)
        
        testPoints=ess.testPoints
        munisf,signisf=nigp.predict(testPoints, return_cov=1)
        musf,sigsf=gp.predict(testPoints,full_cov=1)
        musfTP,sigsfTP=gpTruePos.predict(testPoints,full_cov=1)
        fTrue=trueField(testPoints)
        mumf,sigmf=lin_mf_model.predict(np.hstack((testPoints,2*np.ones((testPoints.shape[0],1)))))
        SIG=lin_mf_model.predict_covariance(np.hstack((testPoints,2*np.ones((testPoints.shape[0],1)))))
        froSIG=np.linalg.norm(np.linalg.inv(SIG)) if Normalize else 1
        frosigsf=np.linalg.norm(np.linalg.inv(sigsf)) if Normalize else 1
        frosignisf=np.linalg.norm(np.linalg.inv(signisf)) if Normalize else 1
        frossigsfTP=np.linalg.norm(np.linalg.inv(sigsfTP)) if Normalize else 1
        #compute weighted mean squared error
        emf=mumf-fTrue
        WMSEmf=np.matmul(np.matmul(emf.T,np.linalg.inv(SIG)/froSIG),emf)/emf.shape[0]
        
        esf=musf-fTrue
        WMSEsf=np.matmul(np.matmul(esf.T,np.linalg.inv(sigsf)/frosigsf),esf)/esf.shape[0]
        
        munisf.shape=(munisf.shape[0],1)
        enisf=munisf-fTrue
        WMSEnisf=np.matmul(np.matmul(enisf.T,np.linalg.inv(signisf)/frosignisf),enisf)/enisf.shape[0]
        
        esfTP=musfTP-fTrue
        WMSEsfTP=np.matmul(np.matmul(esfTP.T,np.linalg.inv(sigsfTP)/frossigsfTP),esfTP)/esfTP.shape[0]
        np.savetxt(os.getcwd()+savePath+file_name.replace("GPData", "GPRes"),np.concatenate((testPoints,fTrue,musf,sigsf,mumf,sigmf),axis=1),delimiter=",",header=" x,y,z,trueField,sfMean,sfVar,mfMean,mfVar",comments="")
        
        errString=""
        RMSE=np.sqrt(np.mean((emf)**2))
        errString=errString+"RMSE mf:{}\n".format(RMSE)
        print("RMSE mf:",RMSE)
        RMSE=np.sqrt(np.mean((esf)**2))
        errString=errString+"RMSE sf:{}\n".format(RMSE)
        print("RMSE sf:",RMSE)
        RMSE=np.sqrt(np.mean((enisf)**2))
        errString=errString+"RMSE nisf:{}\n".format(RMSE)
        print("RMSE nisf:",RMSE)
        RMSE=np.sqrt(np.mean((esfTP)**2))
        errString=errString+"RMSE sfTP:{}\n".format(RMSE)
        print("RMSE sfTP:",RMSE)
        
        print("WRMSE mf:",WMSEmf)
        errString=errString+"WRMSE mf:{}\n".format(WMSEmf)
        print("WRMSE sf:",WMSEsf)
        errString=errString+"WRMSE sf:{}\n".format(WMSEsf)
        print("WRMSE nisf:",WMSEnisf)
        errString=errString+"WRMSE nisf:{}\n".format(WMSEnisf)
        print("WRMSE sfTP:",WMSEsfTP)
        errString=errString+"WRMSE sfTP:{}\n".format(WMSEsfTP)
        
        tempFile=open(os.getcwd()+savePath+file_name.replace("GPData", "MSE").replace(".csv",".txt"),"w")
        tempFile.write(errString)
        tempFile.close()
    