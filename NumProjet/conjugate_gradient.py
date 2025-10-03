#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:50:50 2025

@author: sgardettehofmann
"""

import numpy as np


def conjugate_gradient(A, b, x, tol=None):
    """
    Return the solution to A * x = b using the conjugate gradient method.
    """
    if tol is None:
        tol = np.finfo(b.dtype).eps
    
    # Initialize residual vector
    residual = b - np.dot(A, x)
    
    # Initialize search direction vector
    search_direction = residual.copy()
    
    # Compute initial squared residual norm
    old_resid_norm = np.linalg.norm(residual)
    
    # Iterate until convergence
    while old_resid_norm > tol:
        A_search_direction = np.dot(A, search_direction)
        step_size = old_resid_norm**2 / np.dot(search_direction, A_search_direction)
        
        # Update solution
        x += step_size * search_direction
        
        # Update residual
        residual -= step_size * A_search_direction
        new_resid_norm = np.linalg.norm(residual)
        
        # Update search direction vector
        beta = new_resid_norm**2 / old_resid_norm**2
        search_direction = residual + beta * search_direction
        
        # Update squared residual norm for next iteration
        old_resid_norm = new_resid_norm
    
    return x
