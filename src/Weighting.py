# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:53:34 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

# Default libraries
import math

# Install libraries
import numpy as np
import pytest

# Explicit libraries

def normalize_weights(weights):
    """
    Generates the normalized weight matrix with sum of 1. 

    Parameters
    ----------
    weights : numpy.ndarray
        Weight array, 1-D shape.

    Returns
    -------
    normalised_weights : numpy.ndarray
        Same weight array but normalized to have a sum of 1.

    """
    n_particles = len(weights)
    normalised_weights = np.array([wt/sum(weights) for wt in weights])
    
    return normalised_weights

def calculate_linear_weights(scores, CF=2):
    """
    Calculate linear weights based on scores. Use correction factor to adjust tail length.

    Parameters
    ----------
    scores : numpy.ndarray
        One dimensional list of error values generated from the error function.
    CF : float, optional
        Correction factor, use to adjust the lenght of the tail distribution. The default is 2. 

    Returns
    -------
    weights : numpy.ndarray
        Weights of all particle probabilities.

    """
    N          = len(scores)
    weights    = np.ones(N)/N
    
    # Correction factor
    correction = CF*N
    
    for i, sc in enumerate(scores):
        w = 1-(sc/correction)
        if w < 0:
            w = 1.e-30      # Avoid divide by zero
        weights[i] = w
    
    return weights

def calculate_exp_weights(scores, CF=1):
    """
    Calculate weights using exponential decay of scores. Use correction factor to adjust tail length.

    Parameters
    ----------
    scores : numpy.ndarray
        One dimensional list of error values generated from the error function.
    CF : float, optional
        Correction factor, use to adjust the lenght of the tail distribution. The default is 1 and adjusts the following equation (CF*2/# of particles). 

    Returns
    -------
    weights : numpy.ndarray
        Weights of all particle probabilities.

    """

    # Correction factor
    N          = len(scores)
    weights    = np.ones(N)/N
    correction = CF*N
    
    alpha   = math.log(0.5)/(correction**2)
    
    # Apply weighting function to each score in particle filter
    try:
        for i, sc in enumerate(scores):
            weights[i] = np.exp(alpha*sc**2)
            if weights[i] == 0:        
                weights += 1.e-30  # Avoid divide by zero

    except (TypeError, ValueError) as e:
        print(f"WARNING: Cannot calculate weight for value '{sc}': {e}")
        
    except (OverflowError) as e:
        print(f"WARNING: Overflow for value '{sc}': {e}")
        weights[i] = 0
        
    return weights 
    
def calculate_logistic_weights(scores, CF=1):
    """
    Calculate Gaussian weights based on scores

    Parameters
    ----------
    scores : numpy.ndarray
        One dimensional list of error values generated from the error function.
    CF : float, optional
        Correction factor, use to adjust the lenght of the tail distribution. The default is 1 and adjusts the following equation (CF*2/# of particles). 

    Returns
    -------
    weights : numpy.ndarray
        Weights of all particle probabilities.

    """

    # Empty weight array
    N          = len(scores)
    weights    = np.ones(N)/N
    
    # Correction factor
    correction = CF*N
    
    # Apply weighting function to each score in particle filter
    try:
        for i, sc in enumerate(scores):
            weights[i] = 2 - (2 / (1 + np.exp(-sc/correction)))
            
            if weights[i] == 0:
                weights += 1.e-300  # Avoid divide by zero
    except (TypeError, ValueError) as e:
        print(f"WARNING: Cannot calculate weight for value '{sc}': {e}")
            
    except (OverflowError) as e:
        print(f"WARNING: Overflow for value '{sc}': {e}")
        weights[i] = 0
        
    return weights    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    particle_score  = np.arange(start=0, stop=10, step=0.1)
    exp_weight      = calculate_exp_weights(particle_score, CF=2/len(particle_score))
    linear_weight   = calculate_linear_weights(particle_score, CF=2/len(particle_score))
    logistic_weight = calculate_logistic_weights(particle_score, CF=2/len(particle_score))
    
    # Plotting rmspe
    plt.figure(figsize=(8, 5))
    plt.plot(particle_score, exp_weight, label='Exponential weighting', color='blue')
    plt.plot(particle_score, linear_weight, label='Linear weighting', color='green')
    plt.plot(particle_score, logistic_weight, label='Logistic weighting', color='black')

    plt.xlabel('Arbitrary Error value')
    plt.ylabel('Weight')
    plt.title('Weighting with Arbitrary Particle Error CF=2/N')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
    
