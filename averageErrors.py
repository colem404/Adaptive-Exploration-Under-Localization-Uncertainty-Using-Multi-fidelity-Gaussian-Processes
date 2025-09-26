import numpy as np

headers="filename,RMSE mf,RMSE nisf,RMSE sf,RMSE sfTP,T,WRMSE mf,WRMSE nisf,WRMSE sf,WRMSE sfTP,fieldNum,velVariance".split(",")
d=np.genfromtxt("Data\TrajectoriesAndEstimates\GPResults/results.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")


print(d.dtype.names[1],np.mean(d[d.dtype.names[1]]))
print(d.dtype.names[2],np.mean(d[d.dtype.names[2]]))
print(d.dtype.names[3],np.mean(d[d.dtype.names[3]]))
print(d.dtype.names[4],np.mean(d[d.dtype.names[4]]))

print(d.dtype.names[6],np.mean(d[d.dtype.names[6]]))
print(d.dtype.names[7],np.mean(d[d.dtype.names[7]]))
print(d.dtype.names[8],np.mean(d[d.dtype.names[8]]))
print(d.dtype.names[9],np.mean(d[d.dtype.names[9]]))

noise_lev1=d[d.dtype.names[11]]==0
noise_lev2=d[d.dtype.names[11]]==0.1
noise_lev3=d[d.dtype.names[11]]==0.2

print(d.dtype.names[11],"=",0)
print(d.dtype.names[1],np.mean(d[d.dtype.names[1]][noise_lev1]))
print(d.dtype.names[2],np.mean(d[d.dtype.names[2]][noise_lev1]))
print(d.dtype.names[3],np.mean(d[d.dtype.names[3]][noise_lev1]))
print(d.dtype.names[4],np.mean(d[d.dtype.names[4]][noise_lev1]))

print(d.dtype.names[6],np.mean(d[d.dtype.names[6]][noise_lev1]))
print(d.dtype.names[7],np.mean(d[d.dtype.names[7]][noise_lev1]))
print(d.dtype.names[8],np.mean(d[d.dtype.names[8]][noise_lev1]))
print(d.dtype.names[9],np.mean(d[d.dtype.names[9]][noise_lev1]))


print(d.dtype.names[11],"=",0.1)
print(d.dtype.names[1],np.mean(d[d.dtype.names[1]][noise_lev2]))
print(d.dtype.names[2],np.mean(d[d.dtype.names[2]][noise_lev2]))
print(d.dtype.names[3],np.mean(d[d.dtype.names[3]][noise_lev2]))
print(d.dtype.names[4],np.mean(d[d.dtype.names[4]][noise_lev2]))

print(d.dtype.names[6],np.mean(d[d.dtype.names[6]][noise_lev2]))
print(d.dtype.names[7],np.mean(d[d.dtype.names[7]][noise_lev2]))
print(d.dtype.names[8],np.mean(d[d.dtype.names[8]][noise_lev2]))
print(d.dtype.names[9],np.mean(d[d.dtype.names[9]][noise_lev2]))


print(d.dtype.names[11],"=",0.2)
print(d.dtype.names[1],np.mean(d[d.dtype.names[1]][noise_lev3]))
print(d.dtype.names[2],np.mean(d[d.dtype.names[2]][noise_lev3]))
print(d.dtype.names[3],np.mean(d[d.dtype.names[3]][noise_lev3]))
print(d.dtype.names[4],np.mean(d[d.dtype.names[4]][noise_lev3]))

print(d.dtype.names[6],np.mean(d[d.dtype.names[6]][noise_lev3]))
print(d.dtype.names[7],np.mean(d[d.dtype.names[7]][noise_lev3]))
print(d.dtype.names[8],np.mean(d[d.dtype.names[8]][noise_lev3]))
print(d.dtype.names[9],np.mean(d[d.dtype.names[9]][noise_lev3]))