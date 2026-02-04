#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:13:00 2025

@authors: sgardettehofmann, lmannoni
"""

from domain import *
from jacobi import *
from SOR import *
from conjugate_gradient import *

# Compare jacobi and CG methods
def plot_comparison(results1, results2, iter_max, tol, precision, N_list):
    plt.figure(figsize=(8,6))

    #Log iterations 
    iterations = np.arange(1, iter_max + 1)
    log_iter = np.log10(iterations)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(N_list)))

    for color, N in zip(colors, N_list):
        # --- Method 1 (Jacobi) ---
        res_j = results1[N]["residuals"]
        plt.plot(log_iter, np.log10(res_j),
                 linestyle='-',
                 linewidth=1.2,
                 color=color,
                 label=f"Jacobi (N={N})")

        # --- Method 2 (Conjugate Gradient) ---
        res_cg = results2[N]
        plt.plot(log_iter, np.log10(res_cg),
                 linestyle='--',
                 linewidth=1.2,
                 color=color,
                 label=f"CG (N={N})")

    plt.xlabel("log10(Iteration)")
    plt.ylabel("log10(Normalized Residual L2 norm)")
    plt.title(f"Jacobi VS Conjugate Gradient ({precision}, tol={tol})")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Method & Grid size", loc="best")
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
    
    
    
def measure_time(func, *args, repeat=5, **kwargs):
    import time
    times = []

    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs) 
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.mean(times), np.std(times)


def plot_time_performance(data_dict, title="Computational time scaling"):

    plt.figure(figsize=(8,6))

    for method, values in data_dict.items():
        Ns = sorted(values.keys())
        means = np.array([values[N][0] for N in Ns if values[N] is not None])
        stds  = np.array([values[N][1] for N in Ns if values[N] is not None])
        Ns    = np.array([N for N in Ns if values[N] is not None])

        plt.errorbar(
            Ns,
            means,
            yerr=stds,
            marker='o',
            capsize=4,
            linewidth=2,
            label=method
        )

        #computing p as the slope
        if len(Ns) >= 2:
            p = np.polyfit(np.log(Ns), np.log(means), 1)[0]
            print(f"{method} scaling exponent ≈ {p:.2f}")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Grid size N", fontsize=13)
    plt.ylabel("Time (s)", fontsize=13)
    plt.title(title, fontsize=14)

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.show()
    


def compute_power_law_fits(data, ignore_first=1):
    exponents = {}

    for method, values in data.items():
        N = np.array(sorted(values.keys()), dtype=float)
        times = np.array([values[n][0] for n in N])

        #to ignore small grid size (python overhead)
        N_fit = N[ignore_first:]
        T_fit = times[ignore_first:]

        p, logk = np.polyfit(np.log(N_fit), np.log(T_fit), 1)

        exponents[method] = p

        print(f"{method:6s}  p = {p:.3f}")

    return exponents


#Estimate memory usage for a given solver
def memory_usage(func, N, precision='double'):

    dtype = np.float32 if precision == 'single' else np.float64

    # size of one N^3 array
    array_bytes = np.zeros((N, N, N), dtype=dtype).nbytes

    if func.lower() == 'jacobi':
        n_arrays = 3
    elif func.lower() == 'sor':
        n_arrays = 3
    elif func.lower() == 'cg':
        n_arrays = 5
    else:
        raise ValueError("Unknown method. Choose 'jacobi', 'sor', or 'cg'.")

    return n_arrays * array_bytes



### MAIN ###

#parameters
# N_list=[8,16,32,64]
# N=64 #8, 16, 32, 64, 128
# L=1.
# iter_max = 5000
# precision='double'
# threshold=5e-5

# data_time = {

#     "Jacobi": {
#         8:(1.2288,0.0060),
#         16:(2.0184,0.0097),
#         32:(6.7133,0.1785),
#         64:(74.2043,5.4333)
#     },

#     "OR": {
#         8:(0.0420,0.0036),
#         16:(0.2526,0.0080),
#         32:(3.2420,0.0310),
#         64:(84.5201,6.5081)
#     },

#     "SOR": {
#         8:(0.1053,0.0007),
#         16:(0.9865,0.0151),
#         32:(24.4705,0.3627),
#         64: (694.9943,7.2963)
#     },

#     "CG": {
#         8:(0.00510,0.00023),
#         16:(0.02320,0.00058),
#         32:(0.1875,0.0136),
#         64:(2.9671,0.2982)
#     }
# }

# #functions 
# results_jac = loop_jacobi(N_list, iter_max, precision=precision, tol=threshold, L=L)
# results_CG = loop_CG(N_list, iter_max, tol=threshold, precision=precision, L=L)

# plot_comparison(results_jac, results_CG, iter_max, threshold, precision, N_list)

# mean_t, std_t = measure_time(loop_SOR, N_list, iter_max, w, precision, threshold, L)
# print(mean_t, std_t)

# compute_power_law_fits(data_time)

# for method in ['jacobi', 'sor', 'cg']:
#     memory = [memory_usage(method, N, precision) for N in N_list]
#     plt.loglog(N_list, memory, marker='o', label=method.upper())

# plt.xlabel("Grid size N")
# plt.ylabel("Memory usage [bytes]")
# plt.title("Memory scaling of Poisson solvers")
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()

