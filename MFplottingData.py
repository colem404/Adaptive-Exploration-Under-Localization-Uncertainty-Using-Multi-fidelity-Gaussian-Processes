import numpy as np
import GPy
import emukit.multi_fidelity 
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
import plotResultsSettings as ess

seednum=9
planNum=0
tnum=seednum
fDest="MFE"
folder="Data/simp"+fDest+"/seed"+str(seednum)+"/"
tarFolder="Data/simp"+fDest+"/plottingData/"
#t,x,y,z,fieldMeas,xh,yh,zh,sigx,sigy,sigz
dataArr=np.loadtxt(folder+"mfgpSimSimp.csv",skiprows=1,delimiter=",")
params=np.loadtxt(folder+"emuGP.txt",skiprows=2,delimiter=",")
print(params)
for planNum in range(params.shape[0]-1):
	#t,fidelity
	GPtimes=np.loadtxt(folder+"GPData{0}.csv".format(planNum),skiprows=1,delimiter=",")
	df1=GPtimes[GPtimes[:,1]==0,0].tolist()
	df2=GPtimes[GPtimes[:,1]==1,0].tolist()
	df3=GPtimes[GPtimes[:,1]==2,0].tolist()

	Xhf3=dataArr[np.isin(dataArr[:,0],df3),5:8]
	Xhf2=dataArr[np.isin(dataArr[:,0],df2),5:8]
	Xhf1=dataArr[np.isin(dataArr[:,0],df1),5:8]
	#Xhf3=dataArr[np.isin(dataArr[:,0],df3+df2+df1),5:8]
	#Xhf2=dataArr[np.isin(dataArr[:,0],df2+df1),5:8]
	#Xhf1=dataArr[np.isin(dataArr[:,0],df1),5:8]
	Xf3=dataArr[np.isin(dataArr[:,0],df3),1:4]
	Xf2=dataArr[np.isin(dataArr[:,0],df2),1:4]
	Xf1=dataArr[np.isin(dataArr[:,0],df1),1:4]
	#y3=dataArr[np.isin(dataArr[:,0],df3+df2+df1),4]
	y3=dataArr[np.isin(dataArr[:,0],df3),4]
	y3.shape=(y3.shape[0],1)
	#y2=dataArr[np.isin(dataArr[:,0],df2+df1),4]
	y2=dataArr[np.isin(dataArr[:,0],df2),4]
	y2.shape=(y2.shape[0],1)
	y1=dataArr[np.isin(dataArr[:,0],df1),4]
	y1.shape=(y1.shape[0],1)
	mfXs=[Xf3,Xf2,Xf1]
	mfXhs=[Xhf3,Xhf2,Xhf1]
	mfys=[y3,y2,y1]


	#X_train, Y_train = convert_xy_lists_to_arrays([Xf3, Xf2, Xf1], [y3, y2, y1])
	Xh_train, Y_train = convert_xy_lists_to_arrays(mfXhs,mfys)

	n_fids=3
	kernels = [GPy.kern.RBF(3,ARD=True), GPy.kern.RBF(3,ARD=True),GPy.kern.RBF(3,ARD=True)]
	lik=GPy.likelihoods.Gaussian()
	lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
	#gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train, lin_mf_kernel, n_fidelities=n_fids)
	gpy_lin_mf_model = GPyLinearMultiFidelityModel(Xh_train, Y_train,lin_mf_kernel, likelihood=lik, n_fidelities=n_fids)
	lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, n_fids, n_optimization_restarts=1)
	#lin_mf_model.gpy_model.param_array[:]=params[planNum+1,:]
	lin_mf_model.gpy_model.param_array[:]=params[planNum+1,:15]
	lin_mf_model.set_data(Xh_train,Y_train)


	testPoints=ess.testPoints
	mumf,sigmf=lin_mf_model.predict(np.hstack((testPoints,2*np.ones((testPoints.shape[0],1)))))
	np.savetxt(tarFolder+"results"+str(planNum)+".csv",np.concatenate((testPoints,mumf,sigmf),axis=1),delimiter=",",header=" x,y,z,gpMean,gpVar",comments="")
	print(lin_mf_model.gpy_model.param_array)
	
print(seednum)