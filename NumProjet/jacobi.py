#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:17:46 2025

@author: lmannoni, sgardettehofmann
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from domain import *


### In 1D : 
#linearisation of the 3D vector rho in 1D
def linearize(mat) :
    N=len(mat)
    mat_lin=np.full((N**3), 0.)
    
    for i in range (N) :
        for j in range (N) :
            for k in range (N) :
                mat_lin[i*N**2+j*N+k]=mat[i,j,k]
    #same as mat_lin = mat.flatten()
    return mat_lin

def create_A(N):
    D = np.diag(np.array((N**3),-6))
    L = np.diag(np.array((N**3)-1,1),-1)
    U = np.diag(np.array((N**3)-1,1),-1)

        
### In 3D :
# Discrete Laplacian (periodic boundary condition)
def discrete_laplacian(phi, h):
    
    #laplacian of phi with periodic conditions 
    #returns lap = (sum_neighbours - 6*phi) / h^2
    
    #roll implements periodic BCs
    mat = (
        np.roll(phi,  1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi,  1, axis=1) + np.roll(phi, -1, axis=1) +
        np.roll(phi,  1, axis=2) + np.roll(phi, -1, axis=2)
    )
    lap = (mat - 6.0 * phi) / (h**2)
    return lap


def jacobi(rho, N, iter_max, threshold, L=1.) :
    phi = np.full((N,N,N),0)
    phi_next = np.full((N,N,N),0)
    h=L/N
    residuals = []
    iterations = []
    v=True
    
    for i in range(1,iter_max+1) :
        
        #computing phi_n+1
        phi_next = (
            np.roll(phi,1,0) + np.roll(phi,-1,0) 
            + np.roll(phi,1,1) + np.roll(phi,-1,1)
            + np.roll(phi,1,2) + np.roll(phi,-1,2)
            -rho*h**2)/6
        
        #computing the residual as the L2 norm of rho-lap(phi)
        lap_phi = discrete_laplacian(phi_next,h)
        residual = np.linalg.norm(rho-lap_phi)/(N**1.5) #residuals divided by N**3/2 for normalization
        residual_field = (rho - lap_phi)/N**3 #computing the residual field, normalized by the problem size
        
        residuals.append(residual)
        iterations.append(i)
            
        phi = phi_next
        
        #threshold reached before iter_max
        if (residual<threshold):  #filling the list to have iter_max elements
            residuals += [residual] * (iter_max - i)
            iterations += list(range(i+1, iter_max+1))
            iter_threshold=i
            iter_cv=float('inf') #convergence not reached
            res_cv=float('inf') #not really a minimal residual
            print(f"Threshold reached in {i} iterations for N={N}")
            break
        
        #threshold not reached, but residuals converge
        elif (i>8 and (abs(residuals[i-1]-residuals[i-8]))/residuals[i-8] <1e-2/N ) : #convergence threshold
            if (v) : #v = true if the loop was never entered before
                print(f"Convergence reached in {i} iterations for N={N} at value {np.log10(residuals[i-1])}")
                iter_cv = i #convergence not reached
                res_cv = residuals[i-1] #not really a minimal residual
                iter_threshold=float('inf')
                v = False 
            
    if (residual>threshold and v):
        print(f"Threshold and Convergence not reached in {iter_max} iterations for N={N}")
        iter_cv=float('inf') #convergence not reached
        res_cv=float('inf') #not really a minimal residual
        iter_threshold=float('inf')

        
    return phi_next, residuals, res_cv, iter_cv, iter_threshold, residual_field



def multiplot(N_list,iter_max,precision, threshold):
    iterations = np.arange(1, iter_max+1, 1)
    
    ncols, nrows = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)  #flatten in case of 2D array

    
    for ax, N in zip(axes, N_list):
        rho, x, y, z = set_domain(N,L,precision)
        phi_final, residuals = jacobi(rho,N,iter_max, threshold)

        ax.plot(iterations, np.log10(residuals), marker='+', markersize=3)
        ax.set_xlabel("Iteration")
        ax.set_title(f"N = {N}")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.text(0.04, 0.5, "log10(Normalized Residual L2 norm)", va="center", rotation="vertical")
    fig.suptitle(f"Jacobi Convergence for Different Grid Sizes, {precision} precision", fontsize=14)
    plt.subplots_adjust(hspace=2, wspace=0.4, top=0.2)    
    plt.show()
    
    
def loop_jacobi(N_list, iter_max, precision, tol, L):
    results = {}

    for N in N_list:
        rho, x, y, z = set_domain(N, L, precision)
        phi_final, residuals, res_cv, iter_cv, iter_threshold, residual_field = jacobi(rho, N, iter_max, threshold)

        results[N] = {
            "residuals": residuals,
            "res_cv": res_cv,
            "iter_cv": iter_cv,
            "iter_threshold": iter_threshold,
            "residual_field": residual_field
        }

    return results


def plot_jacobi(results, iter_max, precision, tol):
    plt.figure(figsize=(8,6))
    iterations = np.arange(1, iter_max + 1, 1)
 
    for N, data in results.items():
        residuals = data["residuals"]
        plt.plot(iterations, np.log10(residuals),
                 marker='+', markersize=3, linewidth=1,
                 label=f"N = {N}")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Normalized Residual L2 norm)")
    plt.title(f"Jacobi Convergence for different N (threshold={tol}, {precision} precision)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()


def plot_CV_values(method, results, precision, tol):
    N_list = list(results.keys())
    #res_cv_list is the list of the minimal residuals reached, for each N
    res_cv_list = [results[N]["res_cv"] for N in N_list]
    #iter_cv_list is the list of the number of iterations needed to reach convergence, for each N
    iter_cv_list = [results[N]["iter_cv"] for N in N_list]
 
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.set_xlabel("N")
    ax1.set_ylabel("Nb of iterations to converge", color="tab:blue")
    ax1.plot(N_list, iter_cv_list, marker='o', linestyle='-', color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("log10(minimal residual)", color="tab:red")
    ax2.plot(N_list, np.log10(res_cv_list), marker='s', linestyle='--', color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title(f"{method} convergence for different N (threshold={tol}, {precision})")
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_threshold_values(method, results, precision, tol):
    N_list = list(results.keys())
    iter_threshold_list = [results[N]["iter_threshold"] for N in N_list]

    plt.figure(figsize=(8,6))
    plt.plot(N_list, iter_threshold_list,
             marker='o', linestyle='-', linewidth=2)

    plt.xlabel("N")
    plt.ylabel("Number of iterations to reach threshold")
    plt.title(f"{method} speed for different N (threshold={tol}, {precision})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
### MAIN ###

# #parameters
# N_list=[32,64]
# N=64 #8, 16, 32, 64, 128
# L=1.
# iter_max = 7000
# precision='double'
# threshold=5e-10

# #functions
# results = loop_jacobi(N_list, iter_max, precision=precision, tol=threshold, L=L)
# for N, data in results.items():
#     plot2D(data["residual_field"], label=f"Residual field N={N}", coupe="XY")
# plot_jacobi(results, iter_max, precision=precision, tol=threshold)
# plot_threshold_values('Jacobi', results, precision=precision, tol=threshold)
