# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 18:57:29 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import pandas as pd
import numpy as np
import random
import glob
import os
import sys
import matplotlib.pyplot as plt

from Swarm import Swarm

# ============================================================================
# USER CONFIGURATION
# ============================================================================

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
                                             
# Number of Particles to create
NUM_PARTICLES = 100

# Scoring method
METHOD = "normAbs"

# Valid Scoring methods
# exp = exponential decay
# mse = mean square error
# mDist = mahalanobis distance
# normAbs = Normalized absolute difference

# Folder containing CSV files
CSV_FOLDER = "./datasets/016_Q8_175_5000_short/"
CSV_FILE = "histories_2114.csv"

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

# Range for random index values
INDEX_MIN = 0
INDEX_MAX = 2490  # Assuming 2000 datapoints per file

# Short stop option
SHORT_STOP = 2490

# ============================================================================
# MAIN PROGRAM
# ============================================================================



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
    
    # Create list of Particles
    pt_swarm = Swarm(
        num_particles=NUM_PARTICLES,
        index_range=(INDEX_MIN, INDEX_MAX),
        folder_path=CSV_FOLDER,
        selected_cols=COLS,
        method=METHOD
    )
    
    pt_swarm.set_threshold(0.8)
    pt_swarm.set_population_cut(0.5)

    store_particles = list()
    mean_particles = list()
    std_particles = list()
    
#%% RUN PROGRAM 
    # Iterate over all rows and store each as a dict
    for index, row in df_filtered.iterrows():
        # Convert row to dictionary
        row_dict = row.to_dict()
        
        pt_swarm.predict_all()
        pt_swarm.calculate_score_all(row_dict)
        pt_swarm.repopulate()
        pt_swarm.forward_all()
        
        mean_particles.append(pt_swarm.get_mean_pred(col=COLS, cutoff=0.5))
        store_particles.append(pt_swarm.get_current())
        std_particles.append(pt_swarm.get_std_pred(col=COLS, cutoff=0.5))
        
        # Show progress
        if index%10 == 0:
            print(f"Progression: {index}/{len(df_filtered)}")
            
        # Short stop (remove when finished debugging)
        if index == SHORT_STOP:
            print(f"Short stop engaged")
            break        
    
#%% PRINT OUTPUT
    mean = list()
    std  = list()
    X = np.linspace(0, SHORT_STOP, num=SHORT_STOP+1)
    
    for col in COLS:
        # Plotting mean and standard deviation
        mean = [d[col] for d in mean_particles]
        std  = [d[col] for d in std_particles]
    
        lower_bound = np.array(mean)-np.array(std)
        upper_bound = np.array(mean)+np.array(std)
        
        plt.figure()
        plt.plot(X, mean, color='blue', label='Mean')
        plt.plot(X, df_filtered[col][0:SHORT_STOP+1], c="red", label="True")
        plt.fill_between(X, lower_bound, upper_bound, color="gray", alpha=0.3)
    
        # Labels and legend
        plt.title(f"Line Plot w/ Mean and Std.D for {col} using {METHOD}", fontsize=12)
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plotting pred and true value
        plt.figure()
        plt.scatter(X, df_filtered[col][0:SHORT_STOP+1], c="red", s=5, alpha=0.2)
        plt.scatter(X, mean, c='g', s=5, alpha=0.3)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"Scatter Plot of Particles for {col} using {METHOD}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()