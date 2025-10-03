# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib as plt

### Physical Setup

#parameters :
L=1 #value for the cube dimension
N=8 # starting values for grid size (number of cells in the cube along one axis) : 8,16,32,64
sigma=0.2*L
delta=L/N #step size for each axis

rho=np.full((N,N,N),0) #cube full of zero
x=np.array([delta*(i+1/2) for i in range(N)])
y=np.array([delta*(i+1/2) for i in range(N)])
z=np.array([delta*(i+1/2) for i in range(N)])

#def density(x,y,z) : ##x y and z should be numpy arrays to use .shape
#    return (np.exp(-((x-L/2)**2 + (y-L/2)**2 + (z-L/2)**2)/(2*sigma**2)) + np.random.normal(0, 0.1, x.shape))

#directement ci dessous une def compacte de rho mais cette formule marche 
#pas du tout au vu des dimensions des array x y et z, a vor si on essaye plus tard
#rho=np.array([np.exp(-((x[i]-L/2)**2 + (y[j]-L/2)**2 + (z[k]-L/2)**2)/(2*sigma**2)) + np.random.normal(0, 0.1, x.shape)] for i,j,k in range(N))

for i in range(N):
    for j in range(N):
        for k in range(N):
            rho[i,j,k]=np.exp(-((x[i]-L/2)**2 + (y[j]-L/2)**2 + (z[k]-L/2)**2)/(2*sigma**2)) + np.random.normal(0, 0.1)


### Jacobi Method
