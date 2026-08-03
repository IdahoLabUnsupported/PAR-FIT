# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:54:37 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
# Default libraries
import random
import glob
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

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

# Number of Particles to create
NUM_PARTICLES = 100

# Scoring method *Tested, all wo
SCORING = "rmspe"
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
RESAMP = "stratified"
# Threshold   = limit
# Multinomial = uniform distribution resampling based on CDF weights.
# Systematic  = uses evenly spaced positions in the CDF for resampling.
# Stratified  = separates in even partitions then draws one sample from each partition. 

# Pre-sampling method
PREPROC  = "None"
# ESS = Effective Sample Size

# Folder containing CSV files
CSV_FOLDER = "../../datasets/007_Q2_015_0768_T/"
CSV_FILE = "histories_short_print_444.csv"

# Columns to track
COLS = ["FL1", "FL6", "TA21s1", "TL14s1", "PS1", "PS2"]

# Valid Columns names   
# FL1 = Pump 1 flow rate
# FL6 = Pump 2 flow rate
# TL14s1 = Upper plenum temperature
# TA21s1 = Fuel centerline temperature
# PS1 = Pump 1 pump speed
# PS2 = Pump 2 pump speed
# TFL = Total core flow rate (FL1 + FL6)

REPLACE = 0.1       # Replacement Rate
CUTOFF = 0.25       # Output cutoff 
THRESHOLD = 0.5     # Threshold for replacement

# Range for random index values
INDEX_MIN = 0
INDEX_MAX = 2002  # Assuming 2000 datapoints per file

# ============================================================================
# SETUP THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    # Load the CSV file
    df = pd.read_csv(CSV_FOLDER+CSV_FILE)
    
    print(f"Loaded CSV file: {CSV_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check which columns exist
    available_columns = [col for col in COLS if col in df.columns]
    missing_columns = [col for col in COLS if col not in df.columns]

    if missing_columns:
        print(f"WARNING: Missing columns: {missing_columns}")
    
    if not available_columns:
        print("ERROR: None of the specified columns found")
        exit()
    
    # Filter dataframe to only keep specified columns
    df_filtered = df[available_columns]
    
    # Setup swarm of particles
    print(f"\nCreating {NUM_PARTICLES} random Particles...\n")
    
    # Create list of Particles
    pt_swarm = Swarm(
        num_particles=NUM_PARTICLES,
        folder_path=CSV_FOLDER,
        selected_cols=COLS,
        scoring=SCORING,
        weighting=WEIGHTING,
        repopulate=RESAMP,
        replacement_rate=REPLACE,
        preprocessing=PREPROC
    )
    
    pt_swarm.set_threshold(THRESHOLD)
    pt_swarm.set_population_cut(CUTOFF)

    mean_particles = list()
    std_particles  = list()
    tag_particles  = list()
    idx_particles  = list()   
    
    # Setup SPRT boundaries
    beta        = 0.10     #Significance Level
    alpha       = 0.05     #Significance Level 
    normal_mean = 0
    normal_var  = 10
    bias        = 0
    k_var       = 3
    reset_window= 25
    s_test = SPRT(alpha=alpha, 
                  beta=beta, 
                  normal_mean=normal_mean, 
                  normal_var=normal_var, 
                  bias=bias, 
                  k_var=k_var, 
                  reset_window=reset_window)
    
    outcome_tracker = list()
    
#%% RUN TRANSIENT
    # Iterate over all rows and store each as a dict
    for index, row in df_filtered.iterrows():
        # Convert row to dictionary
        row_dict = row.to_dict()
        
        # Make a prediction on all particles
        pt_swarm.predict()
        
        # Get tracking information for particles
        mean = pt_swarm.get_mean_pred(cutoff=CUTOFF)
        std  = pt_swarm.get_std_pred(cutoff=CUTOFF)
        
        # Track parameters
        mean_particles.append(mean)
        std_particles.append(std)
        
        # Calculate instantenous residual
        result = {key: mean[key] - row_dict[key] for key in mean if key in row_dict}
        residual = list(result.values())
        
        # Calculate SPRT value
        outcome_tracker.append(s_test.calculate_SPRT(residual))
        
        # Calculate the error score for each particle relative to known value
        pt_swarm.calculate_score(row_dict)
        
        # Calculate weight of each error score and normalize to sum to 1
        pt_swarm.calculate_weights()
        
        # Repopulate particles that are poor behaving
        pt_swarm.repopulate()
        
        # Move particle tracking forward by one
        pt_swarm.forward()
        
        # Show progress
        if index%50 == 0:            
            print(f"Progression: {index}/{len(df_filtered)}")
    
#%% PRINT OUTPUT    
    mean = list()
    std  = list()
    X = np.linspace(0, len(df), num=len(df))
    
    for col in COLS:
        # Plotting mean and standard deviation
        mean = [d[col] for d in mean_particles]
        std  = [d[col] for d in std_particles]
    
        lower_bound = np.array(mean)-np.array(std)
        upper_bound = np.array(mean)+np.array(std)
        
        plt.figure()
        plt.plot(X, mean, color='blue', label='Mean')
        plt.plot(X, df_filtered[col], c="red", label="True")
        plt.fill_between(X, lower_bound, upper_bound, color="gray", alpha=0.3)
    
        # Labels and legend
        plt.title(f"Mean and Std.D for {col}; Scoring ({SCORING}); Weighting ({WEIGHTING}); Resample ({RESAMP})", fontsize=10)
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # =====================
    # PLOT SPRT Outcome
    # =====================
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(start=0, stop=len(outcome_tracker)) 
    
    # Plot first series on left Y axis
    ax1.scatter(x, outcome_tracker, color='g', s=5, label='Outcome')
    ax1.set_xlabel('X Label')               
    ax1.set_ylabel('Outcome', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='major', linestyle='--', alpha=0.35)
    
    # Create a second axes sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Plot second series on right Y axis
    df_pred = pd.DataFrame(mean_particles) 
    line2,  = ax2.plot(x, df_pred["TA21s1"], color='k', linewidth=2.5, label='True')
    line3,  = ax2.plot(x, df_filtered["TA21s1"], color='r', linewidth=2.5, label='Measured')
    ax2.set_ylabel('Series B (units)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Optional: tighten layout and save
    plt.title('Dual-Axis Plot (Template)')
    fig.tight_layout()
    # plt.savefig('dual_axis_plot.png', dpi=200)
    plt.show()
        # # Plotting pred and true value
        # plt.figure()
        # plt.scatter(X, df_filtered[col], c="red", s=5, alpha=0.2)
        # plt.scatter(X, mean, c='g', s=5, alpha=0.3)
        # plt.xlabel("Time Step", fontsize=12)
        # plt.ylabel("Value", fontsize=12)
        # plt.title(f"Scatter Plot of Particles for {col} using {METHOD}", fontsize=12)
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
