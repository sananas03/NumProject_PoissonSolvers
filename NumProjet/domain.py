#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 15:00:21 2025

@authors: sgardettehofmann, lmannoni
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0) # fixed random seed for reproducibility

### Physical Setup

# Generate a 3D cubic domain with a Gaussian density profile and added noise
def set_domain(N,L=1.0, precision='double'):
   
    sigma=0.2*L     # Gaussian width
    sigma_g=1       # Noise amplitude (inverse signal-to-noise ratio) from 1 to 10 : from least to most noisy
    h=L/N           # Step size for each axis
    dtype = np.float32 if precision == 'single' else np.float64

    # Cell-centered coordinates
    x=np.array([h*(i+1/2) for i in range(N)], dtype=dtype)
    y=np.array([h*(i+1/2) for i in range(N)], dtype=dtype)
    z=np.array([h*(i+1/2) for i in range(N)], dtype=dtype)
    
    rho=np.full((N,N,N),0.) #cube full of zero
    for i in range(N):
        for j in range(N):
            for k in range(N):
                gaussian = np.exp(-((x[i]-L/2)**2 + (y[j]-L/2)**2 + 
                (z[k]-L/2)**2)/(2*sigma**2)) 
                noise = np.random.normal(0.0, 0.1)
                # rho[i,j,k] = gaussian + noise # simple version 
                rho[i,j,k] = gaussian * (1 + sigma_g*noise) #better version with (signal/noise const)
    
    # for periodic Poisson, require zero mean
    mean_rho = rho.mean()
    rho = rho - mean_rho
    
    return rho, x, y, z

            

### Plot 2D d'une coupe au milieu du cube
def plot2D_domain(N,L=1.0):
    
    rho, x, y, z = set_domain(N,L)
    mid = N // 2 #position de la coupe

    plt.imshow(rho[:, :, mid], extent=[0, L, 0, L], origin='lower', cmap='plasma')
    plt.colorbar(label='Density ρ')
    plt.title(f"Central slice (z = {z[mid]:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


### Plot 3D dune coupe au milieu du cube
def plot3D_domain(N,L=1.0):
    
    rho, x, y, z = set_domain(N,L)
    mid = N // 2

    # On crée les grilles 2D correspondantes
    X, Y = np.meshgrid(x, y, indexing='ij')
    Zslice = rho[:, :, mid]  # coupe au milieu du cube

    # --- tracé 3D ---
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Zslice, cmap='plasma', linewidth=0, antialiased=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('ρ(x,y,z=L/2)')
    ax.set_title("3D surface of the Gaussian density (central slice)")
    fig.colorbar(surf, ax=ax, shrink=0.6, label='ρ')
    plt.show()

#fonction generique de plot 2D
def plot2D(mat, label, coupe="XY"):
    
    N = len(mat)
    mid = N // 2  # central slice

    fig, ax = plt.subplots(figsize=(6,5))

    if coupe == "XY":
        im = ax.imshow(mat[:, :, mid], origin='lower', cmap='plasma')
        ax.set_title(f'{label} XY slice at z = {mid}')
        ax.set_xlabel('x index'); ax.set_ylabel('y index')

    elif coupe == "XZ":
        im = ax.imshow(mat[:, mid, :], origin='lower', cmap='plasma')
        ax.set_title(f'{label} XZ slice at y = {mid}')
        ax.set_xlabel('x index'); ax.set_ylabel('z index')

    elif coupe == "YZ":
        im = ax.imshow(mat[mid, :, :], origin='lower', cmap='plasma')
        ax.set_title(f'{label} YZ slice at x = {mid}')
        ax.set_xlabel('y index'); ax.set_ylabel('z index')

    else:
        raise ValueError("coupe must be 'XY', 'XZ', or 'YZ'")

    fig.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.show()

    
#fonction generique de plot 3D
def plot3D(mat, label, coupe="XY"):
    
    N = len(mat)
    mid = N // 2
    X, Y = np.meshgrid(np.arange(N), np.arange(N))

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    if coupe == "XY":
        Z = mat[:, :, mid]
        ax.set_xlabel('x index'); ax.set_ylabel('y index'); ax.set_zlabel(label)
    elif coupe == "XZ":
        Z = mat[:, mid, :]
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
        ax.set_xlabel('x index'); ax.set_ylabel('z index'); ax.set_zlabel(label)
    elif coupe == "YZ":
        Z = mat[mid, :, :]
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
        ax.set_xlabel('y index'); ax.set_ylabel('z index'); ax.set_zlabel(label)
    else:
        raise ValueError("coupe must be 'XY', 'XZ', or 'YZ'")

    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k')
    ax.set_title(f'{label} {coupe} slice at center')
    fig.colorbar(surf, ax=ax, shrink=0.6, label=label)
    plt.show()


"""
### MAIN ###
N = 8
rho, x, y, z = set_domain(N)
print(rho)

plot2D_domain(N)
plot3D_domain(N)
"""
