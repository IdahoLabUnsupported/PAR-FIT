# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:04:33 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

# Default libraries
import random
import math

# Install libraries
import numpy as np

# Explicit libraries
from Particle import Particle
from Weighting import normalize_weights

def resample_population(swarm, indices, weighting):
    """
    Support function that performs the resampling given the particle indices and weights. 

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.
    indices : numpy.ndarray
        Array of particle indices to be replaced.
    weighting : numpy.ndarray
        Array of weights of all particles.

    Returns
    -------
    None.

    """
    N = swarm.num_particles
    
    # Repopulate with new samples using existing samples only
    for i in indices:
        # Get new particle information
        new_tag = swarm.particles[i].tag
        new_idx = swarm.particles[i].index
        
        # Track information for external validation
        swarm.tags_tracker.append(new_tag)
        swarm.index_tracker.append(new_idx)
            
        # Deep copy to avoid shared references
        copied_particle = Particle(new_tag, new_idx, initial_score=1/N)
             
        # Replace particle with new particle
        swarm.particles[i] = copied_particle    
    
    # Check if replacement rate is active which determines how many new samples to draw from the entire dataset
    if swarm.replacement_rate != 0:
        # Determine how many particles to replace
        size           = math.floor(swarm.replacement_rate * swarm.num_particles)

        # Sort resampled particles by original weight (descending)
        sorted_indices = np.argsort(weighting)
        worst_indices  = sorted_indices[:size]
        
        # Get all possible particle tags and index values
        keys_tag    = list(swarm.tag_dist.keys())
        keys_index  = list(swarm.index_dist.keys())
                
        resample_tags  = np.random.choice(keys_tag, size=size)
        resample_index = np.random.choice(keys_index, size=size) 
          
        # Replace bottom particles with resampled copies of top particles
        for i, new_tag, new_idx in zip(worst_indices, resample_tags, resample_index):
            # Deep copy to avoid shared references
            copied_particle = Particle(new_tag, new_idx, initial_score=1/N)
        
            # Replace particle with new particle
            swarm.particles[i] = copied_particle
    
    return
            
def resample_multinomial(swarm):
    """
    This resampler draws N independent random numbers from a uniform distribution relative to the weights of each particle. 

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.
    
    Raises
    ------
    ValueError
        The number of particles must be a natural number. Weights must be non-negative and finite. 
        
    Returns
    -------
    indices (optional) : numpy.ndarray
        Returns the new list of particle indices after resampling.

    """    
    # Get current weight function, check weight function is normalized.
    weighting = swarm.weights

    # Check for correct shape
    M = swarm.num_particles
    N = weighting.shape[0]
    
    if M == 0 or N == 0: 
        raise ValueError("Number of particles/weights must be a positive integer number.")
    
    # Handle edge cases
    if np.any(weighting < 0) or np.any(~np.isfinite(weighting)):
        raise ValueError("Weights must be non-negative and finite.")
    
    total = sum(weighting)
    
    if round(total,1) != 1.0 or total <= 0:
        print("WARNING: Weight matrix normalization error. Sum of weights is: " + str(sum(weighting)))
        
        try:
            probs = normalize_weights(weighting)
            
        except:
            print("Cannot normalize weight matrix. Defaulting to uniform distribution for resampling")
        
            probs = np.full(N, 1.0/N, dtype=np.float64)
    
    else:
        probs = weighting / total
        
    # Build CDF 
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0 # Ensure exact endpoints to handle numerical drift.
    
    # Draw N new particles to replace existing particles
    rng = np.random.default_rng()
    u = rng.random(N)
    indices = np.searchsorted(cdf, u, side="left")
    
    resample_population(swarm, indices, weighting)

    return indices
    
def resample_systematic(swarm):
    """
    This method resamples particles based on their weights while maintaining diversity in particle values, preventing degeneracy. 

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.

    Raises
    ------
    ValueError
        The number of particles must be a natural number. Weights must be non-negative and finite.

    Returns
    -------
    indices (optional) : numpy.ndarray
        Returns the new list of particle indices after resampling.

    """
    # Get current weight function, check weight function is normalized.
    weighting = swarm.weights
    
    # Check for correct shape
    M = swarm.num_particles
    N = weighting.shape[0]
    
    if M == 0 or N == 0: 
        raise ValueError("Number of particles/weights must be a positive integer number.")
    
    # Handle edge cases
    if np.any(weighting < 0) or np.any(~np.isfinite(weighting)):
        raise ValueError("Weights must be non-negative and finite.")
        
    total = sum(weighting)
    
    if round(total,1) != 1.0 or total <= 0:
        print("WARNING: Weight matrix normalization error. Sum of weights is: " + str(sum(weighting)))
        
        try:
            probs = normalize_weights(weighting)
            
        except:
            print("Cannot normalize weight matrix. Defaulting to uniform distribution for resampling")
        
            probs = np.full(N, 1.0/N, dtype=np.float64)
    
    else:
        probs = weighting / total
        
    # Build CDF 
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0 # Ensure exact endpoints to handle numerical drift.

    # equally spaced positions with a single random offset
    u0 = np.random.uniform(0.0, 1.0 / N)
    positions = u0 + (np.arange(N, dtype=np.float64) / N)

    # find indices: for each position, locate first cumulative >= position
    indices = np.searchsorted(cdf, positions, side='left')
    
    resample_population(swarm, indices, weighting)
    
    return indices
    
def resample_stratified(swarm):
    """
    This method resamples particles from N equal partitions, with one random sample per partition.

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.

    Raises
    ------
    ValueError
        The number of particles must be a natural number. Weights must be non-negative and finite.

    Returns
    -------
    indices (optional) : numpy.ndarray
        Returns the new list of particle indices after resampling.

    """
    # Get current weight function, check weight function is normalized.
    weighting = swarm.weights
    
    # Check for correct shape
    M = swarm.num_particles
    N = weighting.shape[0]
    
    if M == 0 or N == 0: 
        raise ValueError("Number of particles/weights must be a positive integer number.")
    
    # Handle edge cases
    if np.any(weighting < 0) or np.any(~np.isfinite(weighting)):
        raise ValueError("Weights must be non-negative and finite.")
        
    total = sum(weighting)
    
    if round(total,1) != 1.0 or total <= 0:
        print("WARNING: Weight matrix normalization error. Sum of weights is: " + str(sum(weighting)))
        
        try:
            probs = normalize_weights(weighting)
            
        except:
            print("Cannot normalize weight matrix. Defaulting to uniform distribution for resampling")
        
            probs = np.full(N, 1.0/N, dtype=np.float64)
    
    else:
        probs = weighting / total
        
    # Build CDF 
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0 # Ensure exact endpoints to handle numerical drift.

    # Equally spaced positions with a single random offset
    rng = np.random.default_rng()
    positions = (np.arange(N, dtype=np.float64) + rng.random(N)) / N

    # find indices: for each position, locate first cumulative >= position
    indices = np.searchsorted(cdf, positions, side='left')
    
    resample_population(swarm, indices, weighting)
    
    return indices

def resample_threshold(swarm):
    """
    This method resamples particles from N equal partitions, with one random sample per partition.

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.

    Raises
    ------
    ValueError
        The number of particles must be a natural number. Weights must be non-negative and finite.

    Returns
    -------
    None

    """
    
    # Replace if threshold is not met either
    for pt_idx in range(len(swarm.particles)):
        pt_weight = swarm.particles[pt_idx].weight
        
        # Discard and regenerate if weight is less than threshold
        if pt_weight > swarm.threshold:
            # Randomly get a new particle location
            new_tag = random.choice(swarm.tags_list)
            new_idx = random.randint(swarm.index_range[0], swarm.index_range[1])
             
            # Create new particle
            new_particle = Particle(new_tag, new_idx, initial_score=1/swarm.num_particles)
             
            # Replace particle with new particle
            swarm.particles[pt_idx] = new_particle
            
    return
            
def resample_ESS(swarm):
    """
    Check if effective sample size condition is met for resampling

    Parameters
    ----------
    swarm : Swarm
        Instance of Swarm, the particle filter to resample from.

    Returns
    -------
    bool
        Determines if resampling is needed. If true, a resampler algorithm should be used, if false, particles are not resampled.
    Ntest : float
        The test value generated to determine if resampling is needed. Ntest <= N/2 is the specified limit. 

    """
    scores    = swarm.scores
    weighting = swarm.weights
    Neff  = len(scores)/2   # limit to resample
    Ntest = 1/sum(weighting**2)
    
    if Ntest <= Neff:
        return True, Ntest
    
    else:
        return False, Ntest

def resample_RR(swarm):
    """
    Not implemented
    """
    return True

              