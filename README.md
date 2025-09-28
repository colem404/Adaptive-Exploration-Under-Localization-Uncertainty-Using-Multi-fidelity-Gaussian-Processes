# Adaptive-Exploration-Under-Localization-Uncertainty-Using-Multi-fidelity-Gaussian-Processes
exploreSimSettings.py: file for setting simulation parameters

trajectoryEstimateGenerator.py: takes a .csv file with a trajectory (t,x,y,z) and estimates it using a kalman filter. Produces a .csv file with the original trajectory (t,x,y,z), position estimates (xh,yh,zh), variance of position estimate ate each time step (sigx,sigy,sigz), and position errors (xe,ye,ze).

measFieldData.py: takes a .csv file with the original trajectory (t,x,y,z) and produces a .csv file with the trajectoy and and associated field values (t,x,y,z,fieldVal) for a given field.

prepGPData.py: takes a file with t,x,y,z,xh,yh,zh,sigx,sigy,sigz,xe,ye,ze and produces a file with a GP training data (t,x,y,z,xh,yh,zh,fieldVal,fidLev) where fidLevel is the assigned 

fidelity level (based on sig_x and sig_y) for the multi-fidelity GP model.

GPTrainers.py: takes a file with a GP dataset (t,x,y,z,xh,yh,zh,fieldVal,fidLev) and trains each of the GP models
