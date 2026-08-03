# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:39 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

# Default libraries


# Install libraries
import numpy as np

# Explicit libraries
import Scoring

class Particle:
    """
    A class to store information about a data point. Acts as a particle in the filter.
    
    Attributes:
        tag (str): The name of the file
        index (int): The index value
        current (dict): The determined value by this particle
        score (int): The calculated score value
    """
    
    def __init__(self, tag, index, initial_score, weight=0, duration=0):
        """
        Initialize a Particle object.

        Parameters
        ----------
        tag : string
            The name of the file. This is used to find the corresponding data to search from.
        index : int
            The index value to search the file.
        initial_score : float
            The initial score of a particle. Do not use zero, recommend (1/number of particles) to prevent selection errors.
        weight : float, optional
            The current selection probability for this particle. The default is 0 for new particles. Assigned at runtime.

        Returns
        -------
        None.

        """
        self.tag      = tag
        self.index    = int(index)
        self.score    = initial_score
        self.weight   = weight
        self.duration = duration       # How long to remember previous time steps
        self.current  = None           # Assigned at run time by swarm. Does not need to be assigned.
        
    def calculate_score(self, true_value, method="rmspe"):
        """
        Assigns a score to the particle relative to a measured or true value (true_value). Output is for testing only and not used by the particle filter.

        Parameters
        ----------
        true_value : dict
            The sensor value to compare the particle value to.
        method : string, optional
            Determines the type of method to use to calculate score.. The default is "rmspe".

        Returns
        -------
        float (optional)
            The calculated score value based on the selected error function and provided measured or true value.

        """            
        
        # Error handling
        if isinstance(method, str):
            method = method.lower()
            
            if method == "rmspe":
                # Calculate score using mean square error; score must be single numerical value
                self.score = Scoring.calculate_rmspe_score(self.current, true_value)
            
            elif method == "mape":
                # Calculate score using normalized absolute error; score must be single numerical value
                self.score = Scoring.calculate_mape_score(self.current, true_value)         
            
            elif method == "exp":
                # Calculate score using laplacian distance; score must be single numerical value
                self.score = Scoring.calculate_exp_score(self.current, true_value)  
            
            elif method == "logc":
                self.score = Scoring.calculate_logCosh_score(self.current, true_value)  
          
            else:
                print("Error function not selected. Choose from root mean square percent error (rmspe), mean absolute percent error (mape), exponential error (exp), or log cosh error (logc)")
                self.score = -1
        else:
            print("Error function not valid. Choose from root mean square percent error (rmspe), mean absolute percent error (mape), exponential error (exp), or log cosh error (logc)")
            self.score = -1
        
        return self.score
    
    def forward(self, max_idx):
        """
        Moves the particle foward in the same dataset and increment life

        Parameters
        ----------
        max_idx : int
            Maximum index of the reference datafile. If a CSV has 200 rows, then max_idx=200. Assigned automatically by code. 

        Returns
        -------
        None.

        """
        if self.index < max_idx:
            self.index = self.index+1
    
    def __check_shape__(self, true_value):
        """Check if the two arrays are the same shape"""
        if len(self.current) == len(true_value):
            return True
        else:
            return False
        
    def __str__(self):
        """String representation of the Particle."""
        return f"DataPoint(tag='{self.tag}', index={self.index}, score={self.score})"
    
    def __repr__(self):
        """Official string representation of the Particle."""
        return f"DataPoint('{self.tag}', {self.index}, {self.score})"
    
    def get_tag(self):
        """Get the tag value."""
        return self.tag

    def get_index(self):
        """Get the index value."""
        return self.index
    
    def set_tag(self, tag):
        """Set the tag value."""
        self.tag = tag
    
    def set_index(self, index):
        """Set the index value."""
        self.index = int(index)
    
# Example usage
if __name__ == "__main__":      
    # Create a particle object
    point1 = Particle(tag="test.csv", index=245, initial_score=0)
    
    true_dict = {'PS1': 500,
                 'PS2': 500}
    pred_dict = {'PS1': 250,
                 'PS2': 250}
     
    point1.current = pred_dict
           
    print(point1.calculate_score(true_dict))
    print(point1)

