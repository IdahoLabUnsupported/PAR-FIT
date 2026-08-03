# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:21:23 2025

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

def calculate_rmspe_score(pred_value: dict, true_value: dict ):
    '''
    Calculate root mean square percentage error between matching columns in two dictionaries.
    
    Parameters
    ----------
    pred_value : dict
        First dictionary with column names as keys, floats as values.
    true_value : dict
        Second dictionary with column names as keys, floats as values.

    Raises
    ------
    ValueError
        Keys in pred_value and true_vale do not match. Prints error message. 

    Returns
    -------
    float
        Numerical value for root mean square error between matching keys of pred_value and true_value

    '''

    # Ensure keys match
    if pred_value.keys() != true_value.keys():
        raise ValueError("Prediction and truth dictionaries must have the same keys.")
        
    squared_errors = []
    
    # Find common keys between both dictionaries
    common_keys = set(pred_value.keys()) & set(true_value.keys())
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            if true_value[key] != 0:
                squared_error = ((true_value[key] - pred_value[key])/true_value[key]) ** 2
            elif pred_value[key] != 0:
                squared_error = ((true_value[key] - pred_value[key])/pred_value[key]) ** 2
            else: 
                squared_error = ((true_value[key] - pred_value[key])/1E-30) ** 2
            squared_errors.append(squared_error)
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate error for column '{key}': {e}")
            return None
    
    # Return average MSE
    if squared_errors:
        return math.sqrt(sum(squared_errors) / len(squared_errors))
    else:
        return None

def calculate_mape_score(pred_value: dict, true_value: dict):
    '''
    Calculate mean absolute percent error between matching columns in two dictionaries.

    Parameters
    ----------
    pred_value : dict
        First dictionary with column names as keys and values.
    true_value : dict
        Second dictionary with column names as keys and values.

    Raises
    ------
    ValueError
        Keys in pred_value and true_vale do not match. Prints error message.

    Returns
    -------
    float
        Numerical value for mean absolute percent error between matching keys of pred_value and true_value

    '''
    # Ensure keys match
    if pred_value.keys() != true_value.keys():
        raise ValueError("Prediction and truth dictionaries must have the same keys.")
        
    abs_norm_errors = []
    
    # Find common keys between both dictionaries
    common_keys = set(pred_value.keys()) & set(true_value.keys())
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            if true_value[key] != 0:
                ee = (true_value[key] - pred_value[key]) / true_value[key]   
            elif pred_value[key] != 0:    
                ee = (true_value[key] - pred_value[key]) / pred_value[key]
            else:
                ee = (true_value[key] - pred_value[key]) / 1E-30
            
            abs_norm_errors.append(abs(ee))   
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate error for column '{key}': {e}")
            return None
    
    # Return average absolute error
    if abs_norm_errors:
        return sum(abs_norm_errors) / len(abs_norm_errors)
    else:
        return None

def calculate_exp_score(pred_value: dict, true_value: dict, CF=1):
    """
    Calculates exponential error between matching columns in two dictionaries.

    Parameters
    ----------
    pred_value : dict
        First dictionary with column names as keys and floats as values.
    true_value : dict
        Second dictionary with column names as keys and floats as values.
    CF : Correction Factor; Degree of penalty for error function, higher value is more penalizing, optional
        DESCRIPTION. The default is 1. Valid range (0, inf]. No negative values.

    Raises
    ------
    ValueError
        Keys in pred_value and true_vale do not match. Prints error message.

    Returns
    -------
    float
        Numerical value for exponential decay error between matching keys of pred_value and true_value

    """
    # Ensure keys match
    if pred_value.keys() != true_value.keys():
        raise ValueError("Prediction and truth dictionaries must have the same keys.")

    # Find common keys between both dictionaries
    common_keys = set(pred_value.keys()) & set(true_value.keys())
    
    exp_errors = []
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            if true_value[key] != 0:
                ee = abs((true_value[key] - pred_value[key]) / true_value[key])
            elif pred_value[key] != 0:    
                ee = abs((true_value[key] - pred_value[key]) / pred_value[key])
            else:
                ee = abs((true_value[key] - pred_value[key])) / 1E-30
                
            exp_errors.append(1 - np.exp(-ee))   
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate error for column '{key}': {e}")
            return None
    
    # Compute average error
    error = CF*np.sum(exp_errors)/len(pred_value)

    return float(error)        

def calculate_logCosh_score(pred_value: dict, true_value: dict):
    """
    Calculate Log-Cosh error between matching columns in two dictionaries.

    Parameters
    ----------
    pred_value : dict
        First dictionary with column names as keys and floats as values.
    true_value : dict
        Second dictionary with column names as keys and floats as values.

    Raises
    ------
    ValueError
        Keys in pred_value and true_value do not match. Prints error message.

    Returns
    -------
    float
        Numerical value for log cosh error between matching keys of pred_value and true_value

    """
    
    # Ensure keys match
    if pred_value.keys() != true_value.keys():
        raise ValueError("Prediction and truth dictionaries must have the same keys.")

    # Find common keys between both dictionaries
    common_keys = set(pred_value.keys()) & set(true_value.keys())
    
    logC_errors = []
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            if true_value[key] != 0:
                ee = np.log(np.cosh((true_value[key] - pred_value[key])/true_value[key]))
            elif pred_value[key] != 0:    
                ee = np.log(np.cosh((true_value[key] - pred_value[key])/pred_value[key]))
            else:
                ee = np.log(np.cosh((true_value[key] - pred_value[key])/1E-30))

            logC_errors.append(ee)    
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate error for column '{key}': {e}")
            return None
    
    # Compute average error
    error = np.sum(logC_errors)/len(pred_value)

    return float(error)       
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    """
    Test code to visualize the generated error functions in 2D and 3D. 
    """
    # Generate 2D plot with 3 variables.         
    true_dict = {
        'a': 50,
        'b': 50,
        'c': 50
    }
    
    # Generate error values by varying true values from 0 to 1
    x_vals = np.arange(start=1, stop=100, step=1) 
    y_rmspe  = []
    y_mape = []
    y_exp  = []
    y_logC = []
    
    for x in x_vals:
        pred_dict = {
            'a': x,
            'b': x,
            'c': x
        }
        y_rmspe.append(calculate_rmspe_score(pred_dict, true_dict))
        y_mape.append(calculate_mape_score(pred_dict, true_dict))
        y_exp.append(calculate_exp_score(pred_dict, true_dict))
        y_logC.append(calculate_logCosh_score(pred_dict, true_dict))
    
    # Plotting rmspe
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_rmspe, label='Root Mean Square Percent Error', color='blue')
    plt.plot(x_vals, y_mape, label='Mean Absolute Percent Error', color='red')
    plt.plot(x_vals, y_exp, label='Exponential Error', color='green')
    plt.plot(x_vals, y_logC, label='Log Cosh Error', color='orange')
    plt.xlabel('Predicted Value (0 to 1)')
    plt.axvline(x=50, color='black', linestyle='--', label='True Value')
    plt.ylabel('Error')
    plt.ylim([0,1])
    plt.title('Error Functions with 50 as True Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Mean Square Error: {calculate_rmspe_score(pred_dict, true_dict)}")
    print(f"Normalized Absolute Error: {calculate_mape_score(pred_dict, true_dict)}")
    print(f"Exponential Error: {calculate_exp_score(pred_dict, true_dict)}")
    print(f"Log Cosh Error: {calculate_logCosh_score(pred_dict, true_dict)}")
    
    
    # Plot 3D surface error plot         
    true_dict = {
        'a': 50,
        'b': 50
    }
    
    # Generate error values by varying true values from 0 to 1
    x_vals   = np.arange(start=1, stop=100, step=1)
    y_vals   = np.arange(start=1, stop=100, step=1)
    X, Y     = np.meshgrid(x_vals, y_vals)
    
    z_rmspe  = np.zeros([len(x_vals), len(x_vals)])
    z_mape   = np.zeros([len(x_vals), len(x_vals)])
    z_exp    = np.zeros([len(x_vals), len(x_vals)])
    z_logC   = np.zeros([len(x_vals), len(x_vals)])
    
    for x_idx in range(len(X)):
        for y_idx in range(len(Y)):
            pred_dict = {
                'a': X[x_idx, y_idx],
                'b': Y[x_idx, y_idx]
            }
            z_rmspe[x_idx, y_idx] = calculate_rmspe_score(pred_dict, true_dict)
            z_mape[x_idx, y_idx]  = calculate_mape_score(pred_dict, true_dict)
            z_exp[x_idx, y_idx]   = calculate_exp_score(pred_dict, true_dict)
            z_logC[x_idx, y_idx]  = calculate_logCosh_score(pred_dict, true_dict)
        
    fig1  = plt.figure()
    ax1   = fig1.add_subplot(111, projection='3d', title="RMSPE")
    surf1 = ax1.plot_surface(X, Y, z_rmspe,
                           cmap=cm.viridis, 
                           linewidth=0
                          )
    plt.tight_layout()
    
    
    fig2  = plt.figure()
    ax2   = fig2.add_subplot(111, projection='3d', title="MAPE")
    surf2 = ax2.plot_surface(X, Y, z_mape,
                           cmap=cm.viridis, 
                           linewidth=0
                          )
    plt.tight_layout()

    fig3  = plt.figure()
    ax3   = fig3.add_subplot(111, projection='3d', title="EXP")
    surf3 = ax3.plot_surface(X, Y, z_exp,
                           cmap=cm.viridis, 
                           linewidth=0
                          )
    plt.tight_layout()

    fig4  = plt.figure()
    ax4   = fig4.add_subplot(111, projection='3d', title="LOGC")
    surf4 = ax4.plot_surface(X, Y, z_logC,
                           cmap=cm.viridis, 
                           linewidth=0
                          )

    plt.tight_layout()
    plt.show()
    