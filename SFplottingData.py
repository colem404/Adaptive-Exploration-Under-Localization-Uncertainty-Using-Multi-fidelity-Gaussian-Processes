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
fDest="SFE"
folder="Data/simp"+fDest+"/seed"+str(seednum)+"/"
tarFolder="Data/simp"+fDest+"/plottingData/"
#t,x,y,z,fieldMeas,xh,yh,zh,sigx,sigy,sigz
dataArr=np.loadtxt(folder+"mfgpSimSimp.csv",skiprows=1,delimiter=",")
params=np.loadtxt(folder+"GPySFGP.txt",skiprows=2,delimiter=",")
print(params)
for planNum in range(params.shape[0]):
	#t,fidelity
	GPtimes=np.loadtxt(folder+"GPData{0}.csv".format(planNum),skiprows=1,delimiter=",")
	df1=GPtimes[GPtimes[:,1]==0,0].tolist()
	df2=GPtimes[GPtimes[:,1]==1,0].tolist()
	df3=GPtimes[GPtimes[:,1]==2,0].tolist()

	sfXhs=dataArr[np.isin(dataArr[:,0],df1+df2+df3),5:8]
	sfXs=dataArr[np.isin(dataArr[:,0],df1+df2+df3),1:4]
	sfys=dataArr[np.isin(dataArr[:,0],df1+df2+df3),4]
	sfys.shape=(sfys.shape[0],1)
	kernel=GPy.kern.RBF(input_dim=3,ARD=True)
	gp=GPy.models.GPRegression(sfXhs,sfys,kernel)
	gp.param_array[:]=params[planNum,:]
	gp.set_XY(sfXhs,sfys)


	testPoints=ess.testPoints
	musf,sigsf=gp.predict(testPoints)
	np.savetxt(tarFolder+"results"+str(planNum)+".csv",np.concatenate((testPoints,musf,sigsf),axis=1),delimiter=",",header=" x,y,z,gpMean,gpVar",comments="")

	
	print(gp.param_array)
print(seednum)