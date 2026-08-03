# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:08:10 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import math 

import numpy as np 
import matplotlib.pyplot as plt

hypo_labels = {"normal": 1,
               "abnormal": 2,
               "no data": 0,
               "error": -1
              }

# SPRT Test
class SPRT:
    def __init__(self, alpha, beta, normal_mean=0, normal_var=1, bias=0, k_var=1, reset_window=10):
        self.alpha         = alpha
        self.beta          = beta
        self.upper         = math.log((1-self.beta)/self.alpha)
        self.lower         = math.log(self.beta/(1-self.alpha))
        self.mean_normal   = normal_mean
        self.var_normal    = normal_var
        self.mean_abnormal = self.mean_normal+bias
        self.var_abnormal  = self.var_normal*k_var
        self.S0            = 0
        self.ST            = 0
        self.reset_window  = reset_window
        
    def calculate_SPRT(self, residual):
        residual = np.array(residual)
        
        normal   = self.calculate_LC(residual, self.mean_normal, self.var_normal)
        abnormal = self.calculate_LC(residual, self.mean_abnormal, self.var_abnormal)
        S0       = self.calculate_S0(normal=normal, abnormal=abnormal)
        self.S0 += S0
        self.ST += 1
        
        # make a decision about the condition
        if self.S0 < self.lower:
            # Normal mode, reset the running mean and variance
            self.S0 = 0
            self.ST = 0 
            return "Normal"
        
        elif self.S0 > self.upper:
            # Abnormal mode, reset the running mean and variance
            self.S0 = 0 
            self.ST = 0
            return "Abnormal"
            
        elif self.S0 < self.upper and self.S0 > self.lower and self.ST < self.reset_window:       
            # Not enough data, continue collecting more samples
            return "No data"
        
        else:
            # Not enough data but also the window has expired.
            self.S0 = 0 
            self.ST = 0
            return "Reset"
        
    def calculate_LC(self, residual, mean, var):
        LC = math.sqrt(1/(2*math.pi*var)) * np.exp(-1/2 * (np.power((residual-mean),2))/var)
        return LC
        
    def calculate_S0(self, normal, abnormal):
        return np.log(np.sum(abnormal)/np.sum(normal))
    
if __name__ == "__main__":
    # Setup SPRT boundaries
    beta      = 0.10     #Significance Level
    alpha     = 0.05     #Significance Level 
    
    # Setup distribution for normal
    mean_1 = 0
    var_1  = 5
    L1     = np.random.normal(loc=mean_1, scale=var_1, size=100)
    
    # Setup distribution for abnormal
    mean_2 = 0
    var_2  = 25
    L2     = np.random.normal(loc=mean_2, scale=var_2, size=100)

    # Setup residual distribution
    mean_3 = 0
    var_3  = 0.5 
    drift  = 0.2  # Options for sensor drift
    bias   = -10   # Options for bias
    L3     = np.random.normal(loc=mean_3, scale=var_3, size=100)
    L4     = np.copy(L3)
    
    # Experimental signal with drift
    for i in range(len(L3)):
        L4[i]     = L3[i] + drift*i + bias 
        
    # Setup tracking variables
    outcome_tracker = list()
    
    # Setup 
    s_test = SPRT(alpha=alpha, beta=beta, normal_mean=0, normal_var=5, bias=0, k_var=5, reset_window=10)

    # Generate test value
    for i in range(len(L3)):
        residual  = L4[i]             
        outcome_tracker.append(s_test.calculate_SPRT(residual))
    
    # =====================
    # PLOT
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
    line2, = ax2.plot(x, L1, color='k', linewidth=2.5, label='True')
    line3, = ax2.plot(x, L4, color='r', linewidth=2.5, label='Measured')
    ax2.set_ylabel('Series B (units)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Optional: tighten layout and save
    plt.title('Dual-Axis Plot (Template)')
    fig.tight_layout()
    # plt.savefig('dual_axis_plot.png', dpi=200)
    plt.show()

     