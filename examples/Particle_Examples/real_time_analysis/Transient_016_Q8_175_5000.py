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
import math
from celluloid import Camera

# Explicit libraries
from Swarm import Swarm
from SPRT import SPRT

# ============================================================================
# USER CONFIGURATION
# ============================================================================
                                             
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
WEIGHTING = "logistic"
# exp = exponential decay (similar to Gaussian weighting)
# logistic = logistic
# linear = linear triangle

# Resample method
RESAMP = "Systematic"
# Multinomial = uniform distribution resampling based on CDF weights.
# Systematic  = uses evenly spaced positions in the CDF for resampling.
# Stratified  = separates in even partitions then draws one sample from each partition. 
# Threshold   = limit based resampling using weights 

# (Optional) Threshold for replacement
# This value is used a hard replacement criteria. 
if RESAMP.lower() == "threshold":
    THRESHOLD = 0.5

# Pre-sampling method
PREPROC  = "None"
# ESS = Effective Sample Size

# Folder containing CSV files
CSV_FOLDER = "../../datasets/016_Q8_175_5000_trimmed/"

# MODIFY HERE; CHOOSE A CSV FILE
CSV_FILE = "histories_9.csv"

# Columns to track
COLS = ["FL1", "FL6", "TA21s1", "TL14s1", "PS1", "PS2"]

# Valid Columns    
# FL1 = Pump 1 flow rate
# FL6 = Pump 2 flow rate
# TL14s1 = Upper plenum temperature
# TA21s1 = Fuel centerline temperature
# PS1 = Pump 1 pump speed
# PS2 = Pump 2 pump speed
# TFL = Total core flow rate (FL1 + FL6)

# Replacement percentage; 
# Determines the percentage of worst performing particles to randomly replace regardless. Prevents degeneracy of particles.
REPLACE = 0.15

# Cutoff percentatge
# Not all particles are relevant, this value eliminates a percentage of the worst particles from the mean and SPRT calculation. 
CUTOFF = 0.25

# ============================================================================
# RUN THE PROGRAM
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
    
    #folder = CSV_FOLDER
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
    
    if RESAMP.lower() == "threshold":
        pt_swarm.set_threshold(THRESHOLD)
        
    pt_swarm.set_population_cut(CUTOFF)

    mean_particles = list()
    std_particles  = list()
    tag_particles  = list()
    idx_particles  = list()    

    # Setup SPRT boundaries
    beta        = 0.10     # Significance Level
    alpha       = 0.05     # Significance Level 
    normal_mean = 0        # Mean for normal readings
    normal_var  = 10       # Variance for normal readings
    bias        = 0        # Bias for abnormal readings
    k_var       = 3        # Scale of abnormal reading variance; var = k_var*normal_var 
    reset_window= 25       # How many time steps of "No Data" before resetting
    s_test = SPRT(alpha=alpha, 
                  beta=beta, 
                  normal_mean=normal_mean, 
                  normal_var=normal_var, 
                  bias=bias, 
                  k_var=k_var, 
                  reset_window=reset_window)
    
    outcome_tracker = list()
    
#%% RUN PROGRAM 
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
        status = s_test.calculate_SPRT(residual)
        outcome_tracker.append(status)
        
        # Trigger kd_conf if abnormal
        if False:
            track = pt_swarm.report(row_dict)
            kdc_list.append(track)            
        
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
        
        # Stop for future predictions
        future_idx = 2500
        if index >= future_idx:
            break
    
    #future_mean, future_std = pt_swarm.future(timesteps=len(df)-future_idx)
    
    #mean_particles += future_mean
    #std_particles  += future_std
    
#%% PRINT OUTPUT    
    # =====================
    # PLOT Overall Particle Filter Statistics
    # =====================
    tags, indices = pt_swarm.report()
    fig           = plt.figure()
    camera        = Camera(fig)
    x_range       = np.arange(1,2501)
    step          = 10
    plt.xlim((0, 2500))
    plt.ylim((0, 150))
    plt.xlabel("Arbitrary Time")
    plt.ylabel("Count")
    plt.grid()
    
    counts = np.zeros(2500)
    num_chunks = len(indices) // NUM_PARTICLES
    chunks = [indices[i*100:(i+1)*100] for i in range(num_chunks)]
    
    for i in range(num_chunks):
        chunk = chunks[i]
        counts += np.bincount(chunk, minlength=2501)[1:2501]
        if i%step == 0:
            plt.plot(x_range, counts)
            camera.snap()
    
    animation = camera.animate()
    
    # Plot time series prediction
    # Next animation
    mean = [d["TL14s1"] for d in mean_particles]
    std  = [d["TL14s1"] for d in std_particles]
    
    fig1           = plt.figure()
    camera1        = Camera(fig1)
    plt.xlim((0, 2500))
    plt.ylim((420, 480))
    plt.xlabel("Arbitrary Time")
    plt.ylabel("Temperature")
    plt.grid()
    
    chunks_mean = [mean[0:(i+1)*step] for i in range(len(mean))]
    chunks_std  = [std[0:(i+1)*step] for i in range(len(std))]
    
    for i in range(len(mean)):
        chunk_mean = chunks_mean[i]
        chunk_std  = chunks_std[i]
        x_range    = np.arange(0,len(chunk_mean))
        lower_bound = np.array(chunk_mean)-np.array(chunk_std)
        upper_bound = np.array(chunk_mean)+np.array(chunk_std)
        plt.fill_between(x_range, lower_bound, upper_bound, color="gray", alpha=0.3)
        plt.plot(x_range, chunk_mean, color='blue', label='Mean')
        camera1.snap()
    
    animation1 = camera1.animate()
    
    # Plot moving distribution
    fig3           = plt.figure()
    camera3        = Camera(fig3)
    x_range        = np.arange(1,2501)
    plt.xlim((0, 2500))
    plt.ylim((0, 150))
    plt.xlabel("Arbitrary Time")
    plt.ylabel("Count")
    plt.grid()
    
    counts = np.zeros(2500)
    neg    = np.zeros(2500) - np.ones(2500)/2
    num_chunks = len(indices) // NUM_PARTICLES
    chunks = [indices[i*100:(i+1)*100] for i in range(num_chunks)]
    
    for i in range(num_chunks):
        chunk = chunks[i]
        counts += neg
        counts += np.bincount(chunk, minlength=2501)[1:2501]
        counts += np.bincount(chunk, minlength=2501)[1:2501]
        counts = np.maximum(counts, 0)
        if i%step == 0:
            plt.plot(x_range, counts)
            camera3.snap()
    
    animation3 = camera3.animate()
    plt.show()
    
    # UNCOMMENT THE BELOW LINE TO SAVE THE ANIMATION
    #animation.save('./Count.gif')
    #animation1.save('./TL14s1.gif')
    #animation3.save('./Wave.gif')
    
    # =====================
    # PLOT Parametric Outcome
    # =====================
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