#%%
import numpy as np
from PAT import *
from tools import *

# phantom loading
N = 318
u = np.zeros((N,N))
u[N//2,N//2] = 1

# general parameters 
c = 1500 
L = 30e-3
Fs = 50e6 

# sensor parameters
R = 25e-3 
Ntrans = 51 
theta = np.linspace(0, np.pi, Ntrans) 
capteurs = np.zeros((2, 1, Ntrans))

capteurs[0,:,:] = R*np.cos(theta)
capteurs[1,:,:] = R*np.sin(theta)
capteurs[:,:,0] = capteurs[:,:,0] + 25e-3

# parameters 
c = 1500 #m/s
L = 30e-3
Fs = 50e6 
R = 25e-3

# creating matrix
tstart = 5e-3/c
tend = 50e-3/c

A = PAT(capteurs, L, N, c, Fs, tstart, tend)

# saving the sensors
np.savez('sensor_array', capteurs)

# saving the model A
save_sparse_csr('Afwd_318', A)

# %%
