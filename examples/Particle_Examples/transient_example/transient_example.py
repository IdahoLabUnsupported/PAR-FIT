# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:49:26 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
# Default libraries
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../modelTraining')))

# Install libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import torch

# Explicit libraries
from Swarm import Swarm
from SPRT import SPRT
from Model_FNN import FNN

# ============================================================================
# USER CONFIGURATION FOR PARTICLE FILTER
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
WEIGHTING = "logistic"
# exp = exponential decay (similar to Gaussian weighting)
# logistic = logistic
# linear = linear triangle

# Resample method
RESAMP = "Stratified"
# Threshold   = limit
# Multinomial = uniform distribution resampling based on CDF weights.
# Systematic  = uses evenly spaced positions in the CDF for resampling.
# Stratified  = separates in even partitions then draws one sample from each partition. 

# Pre-sampling method
PREPROC  = "None"
# ESS = Effective Sample Size

# Other parameters
REPLACE = 0.1        # Replacement Rate
CUTOFF = 0.35        # Output cutoff 
THRESHOLD = 0.95     # Threshold for replacement

# Folder containing CSV files for particle filter reference
CSV_FOLDER = "../../../datasets/007_Q2_015_0768_T/"

# Columns to track
COLS = ["FL19", "TL14s1", "TA21s1"]

# Valid Columns names   
# FL1 = Pump 1 flow rate
# FL6 = Pump 2 flow rate
# TL14s1 = Upper plenum temperature
# TA21s1 = Fuel centerline temperature
# PS1 = Pump 1 pump speed
# PS2 = Pump 2 pump speed
# TFL = Total core flow rate (FL1 + FL6)

# Create list of Particles
print(f"\nCreating {NUM_PARTICLES} random Particles...\n")
pf_swarm = Swarm(
            num_particles=NUM_PARTICLES,
            folder_path=CSV_FOLDER,
            selected_cols=COLS,
            scoring=SCORING,
            weighting=WEIGHTING,
            repopulate=RESAMP,
            replacement_rate=REPLACE,
            preprocessing=PREPROC
    )
    
pf_swarm.set_threshold(THRESHOLD)
pf_swarm.set_population_cut(CUTOFF)
    
# ============================================================================
# USER CONFIGURATION FOR SPRT 
# ============================================================================

 # Setup SPRT boundaries
alpha       = 0.10     #Significance Level 
beta        = 0.10     #Significance Level
normal_mean = 0
normal_var  = 20
bias        = 0
k_var       = 3
reset_window= 25

# Create model
s_test = SPRT(alpha=alpha, 
              beta=beta, 
              normal_mean=normal_mean, 
              normal_var=normal_var, 
              bias=bias, 
              k_var=k_var, 
              reset_window=reset_window)

# ============================================================================
# USER CONFIGURATION FOR FNN (PREDICTIVE MODEL)
# ============================================================================

# Folder containing trained NN model
FNN_FOLDER = "./models/FNN_007_Q2_015_0768_T_version_1/"
FNN_NAME   = "FNN_007_Q2_015_0768_T_version_1"

engine      = torch.device("cpu")
modelshape  = np.loadtxt(FNN_FOLDER + FNN_NAME + "_Shape.txt", delimiter=',', dtype=int)
FNNmodel    = FNN(modelshape[0], modelshape[1], modelshape[2]).to(engine)
state_dict  = torch.load(FNN_FOLDER + FNN_NAME + ".pt")
FNNmodel.load_state_dict(state_dict)
FNNmodel.eval()
    
# ============================================================================
# USER CONFIGURATION FOR NEW INCOMING DATA
# ============================================================================

# Folder containing new incoming data
TEST_FOLDER = "../../../datasets/015_Q3_016_4096_T/"
TEST_FILE   = "histories_450.csv"

# ============================================================================
# USER CONFIGURATION FOR PLOTTING
# ============================================================================
# Show all plots?
VERBOSE = True

# Range for index values for plotting
INDEX_MIN = 0
INDEX_MAX = 2000  # Assuming 2000 datapoints per file

# ============================================================================
# SETUP TRACKING INFORMATION
# ============================================================================

mean_particles = list()
std_particles  = list()

FNN_output     = list() 
    
outcome_tracker     = list()
future_tracker_mean = list() 
future_tracker_std  = list()
    
if __name__ == "__main__":    
    # Load the test CSV dataset
    df_test = pd.read_csv(TEST_FOLDER+TEST_FILE)
    
    print(f"Loaded CSV file: {TEST_FILE}")
    print(f"Total rows: {len(df_test)}")
    print(f"Columns: {list(df_test.columns)}\n")
    
    # Check which columns exist
    available_columns = [col for col in COLS if col in df_test.columns]
    missing_columns = [col for col in COLS if col not in df_test.columns]
    
    if missing_columns:
        print(f"WARNING: Missing columns: {missing_columns}")
    
    if not available_columns:
        print("ERROR: None of the specified columns found")
        exit()
    
    # Finalize test input dataset
    df_test = df_test[available_columns]
    
    fig    = plt.figure()
    camera = Camera(fig)
    plt.xlim((INDEX_MIN, INDEX_MAX))
    plt.ylim((600, 700))
    plt.grid()
    gnd_truth = df_test["TA21s1"].to_numpy()
    
#%% RUN TRANSIENT
    # Iterate over all rows over the test dataset. Each iteration represents a step.
    for index, row in df_test.iterrows():
        # Format the model input and make a prediction with the FNN model
        dInput      = np.hstack((row["TL14s1"], row["FL19"]))
        input_test  = torch.from_numpy(dInput).to(engine)
        output_pred = (FNNmodel(input_test.float())).detach().numpy()
        output_pred = output_pred.reshape(-1,1)
        FNN_output.append(output_pred.item())
        
        # Add prediction to row for evaluation
        row_pf      = {"FL19": row["FL19"],
                       "TL14s1": row["TL14s1"],
                       "TA21s1": output_pred.item()}
        
        # Make a prediction on all particles 
        pf_swarm.predict()
        
        # Get tracking information for particles
        mean = pf_swarm.get_mean_pred(cutoff=CUTOFF)
        std  = pf_swarm.get_std_pred(cutoff=CUTOFF)
        
        # Calculate instantenous residual
        result = {key: mean[key] - row_pf[key] for key in mean if key in row_pf}
        residual = list(result.values())
        
        # Calculate SPRT value
        outcome_tracker.append(s_test.calculate_SPRT(residual))
        
        # Track parameters
        mean_particles.append(mean)
        std_particles.append(std)
        
        # Calculate the error score for each particle relative to known value
        pf_swarm.calculate_score(row_pf)
        
        # Calculate weight of each error score and normalize to sum to 1
        pf_swarm.calculate_weights()
        
        # Repopulate particles that are poor behaving
        pf_swarm.repopulate()
        
        # Move particle tracking forward by one
        pf_swarm.forward()
        
        # Show progress
        if index%15 == 0:            
            print(f"Progression: {index}/{INDEX_MAX}")
        
            if VERBOSE:                
                # Plotting mean and standard deviation
                mean_curr = [d["TA21s1"] for d in mean_particles]
                std_curr  = [d["TA21s1"] for d in std_particles]
                
                # Predict future steps using particle filter
                ft_mean, ft_std = pf_swarm.future(timesteps=500)
                
                # Forcast on existing predictions 
                mean_future = mean_curr + [d["TA21s1"] for d in ft_mean]
                std_future  = std_curr + [d["TA21s1"] for d in ft_std]
                
                # Determine current uncertainty interval of particles
                lower_bound = np.array(mean_curr)-np.array(std_curr)
                upper_bound = np.array(mean_curr)+np.array(std_curr)
                
                # Determine future uncertainty interval of particles
                lwr_bnd_fut = np.array(mean_future)-np.array(std_future)
                upr_bnd_fut = np.array(mean_future)+np.array(std_future)
                
                # Enforce length of plotting array
                if len(mean_curr)>INDEX_MAX: mean_curr = mean_curr[:INDEX_MAX]
                if len(std_curr)>INDEX_MAX: std_curr = std_curr[:INDEX_MAX]
                
                # Plot for animation
                X     = np.linspace(0, len(mean_curr), num=len(mean_curr))
                X_fut = np.linspace(0, len(mean_future), num=len(mean_future))
                plt.plot(X_fut, mean_future, color="b", linestyle="--", label="Future")
                plt.plot(X, FNN_output, color='r', label="Gnd Truth")
                plt.plot(X, mean_curr, color='k', label='Historical')
                plt.plot(X, gnd_truth[:len(mean_curr)], c="g", label="True")
                plt.fill_between(X_fut, lwr_bnd_fut, upr_bnd_fut, color="gray", alpha=0.3)
                
                camera.snap()
        
        # Early stop 
        if index > INDEX_MAX:
            break
    
    # Stitch and animate the snapshots into gif
    animation = camera.animate()
    plt.show()
    
    # UNCOMMENT THE BELOW LINE TO SAVE THE ANIMATION
    #animation.save('./animation.mp4')
    
#%% Plot OUTPUT    
    
    # =====================
    # PLOT individual outcomes
    # =====================
    
    mean = list()
    std  = list()
    X = np.linspace(0, len(mean_particles), num=len(mean_particles))
    
    for col in COLS:
        # Plotting mean and standard deviation
        mean = [d[col] for d in mean_particles]
        std  = [d[col] for d in std_particles]
    
        lower_bound = np.array(mean)-np.array(std)
        upper_bound = np.array(mean)+np.array(std)
        
        plt.figure()
        plt.plot(X, mean, color='k', label='PF Mean')
        plt.plot(X, df_test[col][:len(mean_particles)], c="r", label="Ground Truth")
        plt.fill_between(X, lower_bound, upper_bound, color="gray", alpha=0.3)
        if col == "TA21s1":
            plt.plot(X, FNN_output, color="g", label="Model Prediction")
    
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
    line3,  = ax2.plot(x, df_test["TA21s1"][:len(df_pred)], color='r', linewidth=2.5, label='Measured')
    ax2.set_ylabel('Series B (units)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Optional: tighten layout and save
    plt.title('Dual-Axis Plot (Template)')
    fig.tight_layout()
    # plt.savefig('dual_axis_plot.png', dpi=200)
    plt.show()

