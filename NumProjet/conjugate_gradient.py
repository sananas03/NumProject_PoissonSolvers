#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:59:41 2025

@authors: sgardettehofmann, lmannoni
"""

import numpy as np
import matplotlib.pyplot as plt
from domain import *
from jacobi import discrete_laplacian


# Solve the 3D Poisson equation using the Conjugate Gradient method
def conjugate_gradient(rho, N, iter_max, tol=1e-8, L=1.0):
    h = L / N
    phi = np.zeros((N, N, N))  # initial guess

    # Initial residual: r₀ = b - Aφ₀ = ρ - ∇²φ₀ = ρ since φ₀ = 0
    r = rho.copy()
    p = r.copy()

    res_norms = [np.linalg.norm(r) / (N ** 1.5)]  # normalisation comme jacobi

    for k in range(iter_max-1):
        Ap = discrete_laplacian(p, h)
        rr_old = np.sum(r * r) # scalar product (r, r)
        alpha = rr_old / np.sum(p * Ap) # step length
        
        # Update solution and residual
        phi = phi + alpha * p
        r = r - alpha * Ap

        rr_new = np.sum(r*r)
        res_norm = np.sqrt(rr_new) / (N ** 1.5)
        res_norms.append(res_norm)

        # Convergence test
        if res_norm < tol:
            print(f"Convergence reached in {k+1} iterations for N={N} at value {np.log10(res_norm)}")
            break

        beta = rr_new / rr_old
        p = r + beta * p
        
        if k == (iter_max-2):
                print(f"Convergence not reached in {iter_max} iterations for N={N}")
                
        # Residual field at convergence (stored at last iteration)    
        residual_field = r.copy()/N**3
        
    return phi, res_norms, residual_field



def loop_CG(N_list, iter_max, tol, precision, L):
    results = {}

    for N in N_list:
        rho, x, y, z = set_domain(N, L, precision)
        phi, residuals, residual_field = conjugate_gradient(rho, N, iter_max, tol, L)

        # Determine minimal residual and iteration where convergence occurs
        res_cv = min(residuals)
        iter_cv = residuals.index(res_cv) + 1  # +1 because iteration start at 1

        # Pad residuals list if it is shorter than iter_max for consistent plotting
        if len(residuals) < iter_max:
            last_res = residuals[-1]
            residuals = residuals + [last_res] * (iter_max - len(residuals))

        # Store results in dictionary
        results[N] = {
            "residuals": residuals,
            "res_cv": res_cv,
            "iter_cv": iter_cv,
            "residual_field": residual_field
        }

    return results


def plot_CG(results, iter_max, tol, precision):
    plt.figure(figsize=(8,6))
    iterations = np.arange(1, iter_max + 1)

    for N, data in results.items():
        residuals = data["residuals"]
        plt.plot(iterations, np.log10(residuals),
                 marker='+', markersize=3, linewidth=1,
                 label=f"N = {N}")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Normalized Residual L2 norm)")
    plt.title(f"Conjugate Gradient Convergence ({precision}, tol={tol})")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Grid size N", loc="best")
    plt.tight_layout()
    plt.show()


#Plot the convergence characteristics
def plot_CV_values(method, results, precision, tol):
    N_list = list(results.keys())
    # List of minimal residuals achieved for each grid size
    res_cv_list = [results[N]["res_cv"] for N in N_list]
    # List of number of iterations needed to reach minimal residual for each grid size
    iter_cv_list = [results[N]["iter_cv"] for N in N_list]
 
    # Plot number of iterations (left y-axis)
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.set_xlabel("N")
    ax1.set_ylabel("Nb of iterations to converge", color="tab:blue")
    ax1.plot(N_list, iter_cv_list, marker='o', linestyle='-', color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Plot minimal residual (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("log10(minimal residual)", color="tab:red")
    ax2.plot(N_list, np.log10(res_cv_list), marker='s', linestyle='--', color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Title + grid
    plt.title(f"{method} convergence for different N (threshold={tol}, {precision})")
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


### MAIN ###

# #parameters
# N_list=[8, 16, 32, 64, 128, 256]
# N=32 
# L=1.0
# iter_max = 1100
# precision='double'
# threshold=5e-16

# #functions
# results = loop_CG(N_list, iter_max, tol=threshold, precision=precision, L=L)
# plot_CG(results, iter_max, tol=threshold, precision=precision)
# plot_CV_values('Conjugate Gradient', results, precision=precision, tol=threshold)

# for N, data in results.items():
#     plot2D(data["residual_field"], label=f"Residual field N={N}", coupe="XY")
