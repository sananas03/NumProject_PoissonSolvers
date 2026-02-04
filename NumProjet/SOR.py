#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:16:43 2025

@authors: sgardettehofmann, lmannoni
"""

from domain import *
from jacobi import *
import math


###3D
###SOR Successive Over Relaxation method : The OR method is implemented first, followed by the SOR method

#Over Relaxation method, with OR parameter w
def OR(rho, N, iter_max, w, threshold=1e-5, L=1.) :
    phi = np.full((N,N,N),0)
    h=L/N #step
    residuals = []
    iterations = []
    v= True 
    
    for i in range(1,iter_max+1) :
        #computing phi_n+1
        phi_next = (1-w)*phi + (
                        np.roll(phi,1,0) + np.roll(phi,-1,0) 
                        + np.roll(phi,1,1) + np.roll(phi,-1,1)
                        + np.roll(phi,1,2) + np.roll(phi,-1,2)
                        -rho*h**2) * w/6

        #computing the residual as the L2 norm of rho-lap(phi) 
        lap_phi = discrete_laplacian(phi_next,h)
        residual = np.linalg.norm(rho-lap_phi)/(N**1.5) #residuals divided by N**3/2 for normalization
        residual_field = (rho - lap_phi)/N**3 #computing the residual field, normalized by the problem size

        residuals.append(residual)
        iterations.append(i)

        
        phi=phi_next
       
        #threshold reached before iter_max
        if (residual<threshold): 
            residuals += [residual] * (iter_max - i) #filling the list to have iter_max elements
            iterations += list(range(i+1, iter_max+1))
            iter_threshold = i
            iter_cv=float('inf') #convergence not reached
            res_cv=float('inf') #not really a minimal residual
            print(f"Threshold reached in {i} iterations for N={N}")
            break
        
        #threshold not reached, but residuals converge
        elif (i>8 and (abs(residuals[i-1]-residuals[i-8]))/residuals[i-8] <5e-2/(N**2) ) : #convergence threshold
            if (v) : #v = true if the loop was never entered before
                print(f"Convergence reached in {i} iterations for N={N} at value {np.log10(residuals[i-1])}")
                iter_cv = i #convergence reached
                res_cv = residuals[i-1] 
                iter_threshold = float('inf') #threshold not reached
                v = False 
             
        #convergence and threshold not reached
    if (residual>threshold and v):
        print(f"Threshold and Convergence not reached in {iter_max} iterations for N={N}")
        iter_cv=float('inf') #convergence not reached
        res_cv=float('inf') #no minimal residual
        iter_threshold = float('inf') #threshold not reached
        
    return phi_next, residuals, res_cv, iter_cv, iter_threshold, residual_field
       
        


def loop_OR(N_list, iter_max, w, precision, threshold, L):
    results = {}

    for N in N_list:
        rho, x, y, z = set_domain(N, L, precision=precision)
        phi_final, residuals, res_cv, iter_cv, iter_threshold,residual_field = OR(rho,N,iter_max,w,threshold=threshold)

        results[N] = {
            "residuals": residuals,
            "res_cv": res_cv,
            "iter_cv": iter_cv,
            "iter_threshold" : iter_threshold,
            "residual_field" : residual_field
        }

    return results



def plot_OR(results, w, iter_max, precision, threshold):
    plt.figure(figsize=(8,6))
    iterations = np.arange(1, iter_max + 1, 1)
 
    for N, data in results.items():
        residuals = data["residuals"]
        plt.plot(iterations, np.log10(residuals),
                 marker='+', markersize=3, linewidth=1,
                 label=f"N = {N}")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Normalized Residual L2 norm)")
    plt.title(f"OR Convergence for different N, w = {w}, (threshold={threshold}, {precision} precision)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()
    
    
    

def plot_w(N_list, iter_max,w_list,threshold,precision='double'): #to plot iter_cv with respect to the parameter w

    for N in N_list:
        iter_threshold_list=[] 
        rho, x, y, z = set_domain(N, L, precision)
        for w in w_list :
            phi_final, residuals, iter_threshold = OR(rho, N, iter_max, w, threshold=threshold)
            iter_threshold_list.append(iter_threshold)
        print(iter_threshold_list)
        plt.plot(w_list,iter_threshold_list, marker='+', markersize=3, linewidth=1, label=f"N={N}")
    
    plt.xlabel("Relaxation Parameter")
    plt.ylabel("Number of iterations to reach threshold")
    plt.title(f"OR speed for Different w ({precision} precision, threshold={threshold})")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()
    

#SOR : adding the succesive part in the OR method
    
def SOR(rho, N, iter_max, w, threshold=1e-5, L=1.):
    phi = np.zeros((N,N,N))
    h = L / N
    residuals = []
    iterations = []
    v=True
    

    
    #plt.figure(figsize=(6,4))
    for i in range(iter_max):

        for j in range(N):
            for k in range(N):
                for l in range(N):

                    phi[j,k,l] = (1-w)*phi[j,k,l] + w/6 * (
                        phi[(j+1)%N, k, l] + phi[(j-1)%N, k, l] +
                        phi[j, (k+1)%N, l] + phi[j, (k-1)%N, l] +
                        phi[j, k, (l+1)%N] + phi[j, k, (l-1)%N] -
                        rho[j,k,l]*h**2)
                    

        #computing the residual as the L2 norm of rho-lap(phi) 
        lap_phi = discrete_laplacian(phi, h)
        residual = np.linalg.norm(rho - lap_phi) / (N**1.5) #residuals divided by N**3/2 for normalization
        residual_field = (rho - lap_phi)/N**3 #computing the residual field, normalized by the problem size

        residuals.append(residual)    
        iterations.append(i)
        
        
        if (residual<threshold): 
            residuals += [residual] * (iter_max - i)
            iterations += list(range(i+1, iter_max+1))
            iter_threshold = i
            iter_cv=float('inf') #convergence not reached
            res_cv=float('inf') #not really a minimal residual
            print(f"Threshold reached in {i} iterations for N={N}")
            break
        
        #threshold not reached, but residuals converge
        elif (i>8 and (abs(residuals[i-1]-residuals[i-8]))/residuals[i-8] <5e-2/N) : #convergence threshold
            if (v) : #v = true if the loop was never entered before
                print(f"Convergence reached in {i} iterations for N={N} at value {np.log10(residuals[i-1])}")
                iter_cv = i #convergence reached
                res_cv = residuals[i-1] 
                iter_threshold = float('inf') #threshold not reached
                v = False 
             
            
    if (residual>threshold and v):
        print(f"Threshold and Convergence not reached in {iter_max} iterations for N={N}")
        iter_cv=float('inf') #convergence not reached
        res_cv=float('inf') #not really a minimal residual
        iter_threshold = float('inf') #threshold not reached
        
    return phi, residuals, res_cv, iter_cv, iter_threshold, residual_field



def loop_SOR(N_list, iter_max, w, precision, threshold, L):
    results = {}

    for N in N_list:
        rho, x, y, z = set_domain(N, L, precision=precision)
        phi_final, residuals, res_cv, iter_cv, iter_threshold, residual_field = SOR(rho,N,iter_max,w,threshold=threshold)

        results[N] = {
            "residuals": residuals,
            "res_cv": res_cv,
            "iter_cv": iter_cv,
            "iter_threshold" : iter_threshold,
            "residual_field" : residual_field
        }

    return results



def plot_SOR(results, w, iter_max, precision, threshold):
    plt.figure(figsize=(8,6))
    iterations = np.arange(1, iter_max + 1, 1)
 
    for N, data in results.items():
        residuals = data["residuals"]
        plt.plot(iterations, np.log10(residuals),
                 marker='+', markersize=3, linewidth=1,
                 label=f"N = {N}")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Normalized Residual L2 norm)")
    plt.title(f"SOR Convergence for different N, w = {w}, (threshold={threshold}, {precision} precision)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()
    
    

def plot_w_SOR(N_list, iter_max,w_list,threshold,precision='double'): #to plot iter_cv with respect to parameter w

    for N in N_list:
        iter_threshold_list=[] 
        rho, x, y, z = set_domain(N, L, precision)
        for w in w_list :
            phi_final, residuals, res_cv, iter_cv, iter_threshold = SOR(rho, N, iter_max, w, threshold=threshold)
            iter_threshold_list.append(iter_threshold)
        print(iter_threshold_list)
        plt.plot(w_list,iter_threshold_list, marker='+', markersize=3, linewidth=1, label=f"N={N}")
    
    plt.xlabel("Relaxation Parameter")
    plt.ylabel("Number of iterations to reach threshold")
    plt.title(f"OR speed for Different w ({precision} precision, threshold={threshold})")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()


### MAIN ###

# #parameters
# N_list=[8, 16, 32, 64, 128]
# N=16
# L=1.
# iter_max = 20000
# precision='double'
# w=0.95
# #w=2/(1+math.sin(math.pi/N))
# #w_list=np.round(np.linspace(0.5,1,5),2)
# threshold=5e-20
# w_list = np.linspace(w- 0.4/2, w + 0.4/2, 15) 


# #functions
# plot_w_SOR(N_list, iter_max,w_list,threshold)
# results = loop_OR(N_list, iter_max, w, precision, threshold, L)
# plot_CV_values(SOR, results, precision, threshold)
# plot_SOR(results, w, iter_max, precision, threshold)
# for N, data in results.items():
#     plot2D(data["residual_field"], label=f"Residual field N={N}", coupe="XY")
