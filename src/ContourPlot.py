# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import matplotlib.pyplot as plt

'''
Description:
    This file does the following:
        - Plots a 2D contour plot
'''
def plot_contour(x,y,z, training_x=[], training_y=[], loc="lower right", xlim=[0,0], ylim=[0,0], title="", xlabel="Feature 1", ylabel="Feature 2",colorbar="Reliability of Prediction"):
    fig2,ax2 = plt.subplots()
    
    # Contour plot with filled levels of reliability
    plt.contourf(x, y, z, zorder=1)
    
    # Training data points
    ax2.scatter(training_x, training_y, color="red", s=1, zorder=2, label="Training input points")
    
    # Options
    ax2.set_axisbelow(True)
    ax2.grid(zorder=1)
    plt.colorbar(label=colorbar)  # Add a colorbar to a plot
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
   
    plt.title(title)
    plt.legend(loc=loc)
    
    return fig2, ax2