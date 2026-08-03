# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:16:22 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

# Default libraries
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Install libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Explicit libraries
from Swarm import Swarm
from SPRT import SPRT

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# VERBOSE
VERBOSE = False
                                            
# Number of Particles to create
NUM_PARTICLES = 100

# Scoring method *Tested, all wo
SCORING = "logc"
# Valid Scoring methods
# exp = exponential decay
# rmspe = root mean square percent error
# mape = mean absolute percent error
# logc = log cosh error

# Weight method *Tested, all work
WEIGHTING = "exp"
# exp = exponential decay (similar to Gaussian weighting)
# logistic = logistic
# linear = linear triangle

# Resample method
RESAMP = "multinomial"
# Multinomial = uniform distribution resampling based on CDF weights.
# Systematic  = uses evenly spaced positions in the CDF for resampling.
# Stratified  = separates in even partitions then draws one sample from each partition. 
# Threshold   = limit based resampling using weights 

# (Optional) Threshold for replacement
# This value is used a hard replacement criteria.
THRESHOLD = 0.5

# Pre-sampling method
PREPROC  = "ESS"
# ESS = Effective Sample Size

# Folder containing CSV files
CSV_FOLDER = "./pf_reference"

# Columns to track
COLS = ["Value1", "Value2", "Value3"]

# Replacement percentage
# Determines the percentage of worst performing particles to randomly replace regardless. Prevents degeneracy of particles.
REPLACE = 0.20

# Cutoff percentage
# Not all particles are relevant, this value eliminates a percentage of the worst particles from the mean and SPRT calculation. 
CUTOFF = 0.3

pf_swarm = Swarm(
			num_particles=NUM_PARTICLES,
			folder_path=CSV_FOLDER,
			selected_cols=COLS,
			scoring=SCORING,
			weighting=WEIGHTING,
			repopulate=RESAMP,
			replacement_rate=REPLACE,
			preprocessing=PREPROC,
			threshold=THRESHOLD,
			population_cut=CUTOFF
)

# Setup SPRT parameters
ALPHA        = 0.05
BETA         = 0.10
normal_mean  = 0
normal_var   = 10
bias         = 0
k_var        = 3
reset_window = 25
pf_SPRT  = SPRT(alpha=ALPHA,
			    beta=BETA,
			    normal_mean=normal_mean,
			    normal_var=normal_var,
			    bias=bias,
			    k_var=k_var,
			    reset_window=reset_window
		)
    
if __name__ == "__main__":
    # Mock measured incoming signal
    signal_part1 = np.random.normal(loc=48,scale=1, size=(100,3))
    signal_part2 = np.random.normal(loc=24,scale=1, size=(100,3))
    measured_signal = np.vstack((signal_part1,signal_part2))
    measured_signal = pd.DataFrame(columns=COLS, data=measured_signal)
    
    # Implement tracking information
    mean_particles  = list()
    std_particles   = list()
    tag_particles   = list()
    idx_particles   = list() 
    SPRT_outcome    = list()
    
    for _, row in measured_signal.iterrows():
        # Convert row to dictionary
        row = row.to_dict()
        
        # Make a prediction on all particles
        pf_swarm.predict()
        
        # Get tracking information for particles
        mean = pf_swarm.get_mean_pred(cutoff=CUTOFF)
        std  = pf_swarm.get_std_pred(cutoff=CUTOFF)
        mean_particles.append(mean)
        std_particles.append(std)
        
        residual = list({key: mean[key] - row[key] for key in mean if key in row}.values())
        
        # Calculate SPRT value
        decision = pf_SPRT.calculate_SPRT(residual)
        SPRT_outcome.append(decision)
        
        # Calculate the error score for each particle relative to known value
        pf_swarm.calculate_score(row)
        
        # Calculate weight of each error score and normalize to sum to 1
        pf_swarm.calculate_weights()
        
        # Repopulate particles that are poor behaving
        pf_swarm.repopulate()
        
        # Move particle tracking forward by one
        pf_swarm.forward()
        
        # Print Results
        if VERBOSE:
            print(f"The particle prediction mean is: {mean}")
            print(f"The particle prediction std is: {std}")
            print(f"The signal is {decision}")
    
    # Plot Particle Filter Outcome
    # Plotting mean and standard deviation
    mean = [d["Value3"] for d in mean_particles]
    std  = [d["Value3"] for d in std_particles]
    X = np.linspace(0, len(measured_signal), num=len(measured_signal))
        
    lower_bound = np.array(mean)-np.array(std)
    upper_bound = np.array(mean)+np.array(std)
    
    plt.figure()
    plt.plot(X, mean, color='blue', label='Mean')
    plt.plot(X, measured_signal["Value3"], c="red", label="True")
    plt.fill_between(X, lower_bound, upper_bound, color="gray", alpha=0.3)
    
    # Labels and legend
    plt.title(f"Mean and Std.D for Value3; Scoring ({SCORING}); Weighting ({WEIGHTING}); Resample ({RESAMP}); Cutoff ({CUTOFF})", wrap=True, fontsize=10)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot SPRT outcome
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(start=0, stop=len(SPRT_outcome)) 
    
    # Plot first series on left Y axis
    ax1.scatter(x, SPRT_outcome, color='g', s=5, label='Outcome')
    ax1.set_xlabel('X Label')               
    ax1.set_ylabel('SPRT Outcome', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='major', linestyle='--', alpha=0.35)
    
    # Create a second axes sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Plot second series on right Y axis
    line2,     = ax2.plot(x, pd.DataFrame(mean_particles)["Value3"], color='k', linewidth=2.5, label='True')
    line3,     = ax2.plot(x, measured_signal["Value3"], color='r', linewidth=2.5, label='Measured')
    ax2.set_ylabel('Value3', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.tight_layout()
    plt.show()