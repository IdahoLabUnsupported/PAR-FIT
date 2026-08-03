# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:50:00 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
# Default libraries
import os
import random
import sys

# Install libraries
import numpy as np
import pandas as pd
import numbers

# Explicit libraries
from Particle import Particle
from Weighting import *
from Resample import *
import kd_conf

# ============================================================================
# MAIN PROGRAM
# ============================================================================
class Swarm:
    def __init__(self, num_particles, 
                 selected_cols=None, 
                 folder_path=None, 
                 data=None, 
                 index_range=None, 
                 scoring="mape", 
                 weighting="exp", 
                 repopulate="multinomial",
                 preprocessing=None,
                 replacement_rate=0.2,
                 threshold=0.5,
                 population_cut=0.75,
                 bound=None
                 ):
        """
        Swarm implements a particle filter algorithm specifically for tracking in time-series data. The filter uses the dataset as a reference to conduct a search. When
        a value is provided, the particle filter will attempt to find the closest dataset and index within that dataset to match. 
        
        Parameters
        ----------
        num_particles : Integer
            Number of particles to be generated in the filter.
        selected_cols : List of strings, optional
            A list of strings that correspond to the column names of the data files. For example ["FL1", "FL6"]. The default is None.
        folder_path : String
            If a folder path is provided, then all csv files in the path are loaded into the reference dataset of the particle filter. The default is None.
        data : Pandas dictionary
            If a data source is specified, then folder path is ignored and the data is used directly. The default is None.
        index_range : tuple, optional
            A tuple containing the number of rows within each csv to consider. The default is None and all rows are considered.
        scoring : String, optional
            The error scoring method to detmine how accurate a particle is. The default is "mape".
        weighting : String, optional
            The method to weigh scores into a normalized array with sum of one. The default is "exp".
        repopulate : String, optional
            The method to repopulate the particles based on particle weight. The default is "multinomial".
        preprocessing : String, optional
            The method to conduct preprocessing prior to resampling as needed. 
        replacement_rate : Float, optional
            When the number of particles is less than the dataset size, this value determines how many new particles should be drawn from the entire dataset. A value of 0 (not recommended) suggests no particles from the entire population are drawn; only the particles that are currently drawn can repopulate the filter. The default is 0.2.
        threshold : Float, optional
            Option for the threshold repopulation method. If the weight is less than the threshold, the particle is replaced. Otherwise unused. The default is 0.5.
        population_cut : Float, optional
            Option for showing results. Not all particle performance is equivalent, poor performing particles introduce noise to the result. This value specifies what percentage of the best performing particles should be used as the output. The default is 0.75.

        Returns
        -------
        None.

        """
        self.particles        = []
        self.num_particles    = num_particles    # Number of partices in the filter
        self.selected_cols    = selected_cols    # Column names in the dataset. Column names are assumed to be the same.
        self.csv_dict         = None             # Location where data is loaded, once 
        
        # Method Options
        self.score_method     = scoring.lower()       # Sets the error scoring method
        self.weight_method    = weighting.lower()     # Sets the weighting method based on error score
        self.repop_method     = repopulate.lower()    # Sets the resampling method
        self.preprocessing    = preprocessing.lower() # Sets the preprocessing method for resampling
        
        # Parameters 
        self.set_threshold(threshold)               # Manually set replacement rate based on particle weight
        self.set_population_cut(population_cut)     # Not all particles are relevant; use only the top X particles based on weight 
        self.set_replacement_rate(replacement_rate) # When resampling, a percentage X of particles are resampled from the entire population. 
        
        # Data storage for available sampling values
        self.tags_list        = list()          # List of active tags in the particle population
        self.index_list       = list()          # List of active indices in the particle population
        self.score_list       = list()          # (Unused) List of active score data in particle population
        self.weights_list     = list()          # (Unused) List of active weights in particle population 
        
        # Tracking parameters for external validation 
        self.tags_tracker     = list()
        self.index_tracker    = list()

        # Instantenous tracking data 
        self.tags             = np.full(num_particles, '', dtype=str)  # Initialize empty string array 
        self.indices          = np.ones(num_particles)                 # Initialized ones array 
        self.scores           = np.ones(num_particles)/num_particles   # Initialize score array, equal score for all particles 
        self.weights          = np.ones(num_particles)/num_particles   # Initialize weight array, equal weight for all particles
        
        # # Parameters for kd_conf
        # self.nn               = 10              # Default value for nearest neighbor calculation
        # self.bound_param      = bound           # Numpy array for boundaries of parameters. 
        
        self.tag_dist         = {}              # (Unused) For tracking the distribution of tags that are called
        self.index_dist       = {}              # (Unused) For tracking the distribution of indices that are called
        
        # A folder path was provided to load the data from
        self.__init_source_data__(folder_path=folder_path, data=data, selected_cols=selected_cols)
        
        # Initializes the valid range and distribution of the swarm
        self.__index_range_finder__(index_range, find_range=True)   
        self.__init_distribution__()
                
        # Assign random values to the particles
        for i in range(num_particles):
            # Randomly select a tag from the list (potentially add latin hypercube sampling here)
            random_tag    = random.choice(self.tags_list)
            self.tags[i]  = random_tag
            
            # Generate random index
            min_range_idx   = self.index_range[0]
            max_range_idx   = self.index_range[1]
            random_index    = random.randint(min_range_idx, max_range_idx)
            self.indices[i] = random_index
            
            # Create particles and add to list
            pt = Particle(random_tag, random_index, initial_score=1)
            self.particles.append(pt)
        
        # # Initialize kd_conf
        # self.KDC = kd_conf.KDC(dSet=self.csv_dict)
        # self.KDC.mod_ext_radius(self.bound_param)
        # self.KDC.mod_neighbors(self.nn)
        
        return

#%% Primary Functions
    def predict(self):
        """
        Get data from each particle using the source data.

        Returns
        -------
        pt_predict : List of dictionary
            This is output prediciton of every particle from the filter as a dictionary. The dictionary contains the column name and the value. Optional return object. Particles are updated in place.

        """
        pt_predict = list()
        for pt in self.particles:
            pt.current = self.__load_data__(pt.get_tag(), pt.get_index())
            pt_predict.append(pt.current)
        
        return pt_predict
    
    def forward(self):
        """
        Move each particle index forward by 1. If the max index is reached, do nothing and keep the current index.

        Returns
        -------
        None.

        """
        for pt in self.particles:
            max_range_idx = self.index_range[1]   # Sets wraparound limit for forwarding
            pt.forward(max_range_idx)
        
    def calculate_score(self, true_value):
        """
        Score the error of every single Particle object based on the true_value reference.

        Parameters
        ----------
        true_value : dict
            The measured or true value to compare the particles against.

        Returns
        -------
        None.

        """
        for i, pt in enumerate(self.particles):
            self.scores[i] = pt.calculate_score(true_value, self.score_method)   
        
    def calculate_weights(self):
        """
        Calculate the weight of each Particle after scoring.

        Returns
        -------
        None.

        """     
        scores = [pt.score for pt in self.particles]
        
        if self.weight_method == "exp":
            weights = calculate_exp_weights(scores)
        elif self.weight_method == "linear":
            weights = calculate_linear_weights(scores)
        elif self.weight_method == "logistic":
            weights = calculate_logistic_weights(scores)
        else: 
            print("WARNING: Weights calculation method not recognized, defaulting to linear method.")
            weights = calculate_linear_weights(scores)
        
        self.weights = normalize_weights(weights)
    
    def repopulate(self):
        """
        Repopulates the particle filter.
        
        Returns
        -------
        None.

        """
        # Calculate the weight of all particles
        scores  = self.scores
        weights = self.weights
        
        # Determine preprocessing resampling need
        if self.preprocessing == "ess":
            # Use ESS to determine if resampling is needed
            resample, _ = resample_ESS(self)
        
        elif self.preprocessing == "rr":
            # User residual resampler to determine where resampling is needed
            resample = resample_RR(self)
        
        else:
            # No preprocessing resampler selected.
            resample = True
            
        # Select resampling method
        if self.repop_method   == "multinomial" and resample:
            resample_multinomial(self)
        
        elif self.repop_method == "systematic" and resample:
            resample_systematic(self)
        
        elif self.repop_method == "stratified" and resample:
            resample_stratified(self)
            
        elif self.repop_method == "threshold":
            # Simplest method to perform resampling; all particles that have a weight less than the threshold are randomly replaced (with replacement) 
            if self.threshold == 0:
                print("WARNING: Threshold resampling method must have a non-zero threshold between 0 and 1.")
            self.swarm = resample_threshold(self)
        
        elif not resample:
            pass  # Do nothing, resampling not needed.
        
        else:
            print("WARNING: Resampling method not recognized, defaulting to multinomial.")
    
    def report(self):
        """
        Determines the quantitative value for reliability based on instantenous proximity. Must be called.
        
        Parameters
        ----------

        Returns
        -------
        
        """    
        return self.tags_tracker, self.index_tracker
        
    def future(self, timesteps=0):
        """
        Calculates future values using the current population of particles. No resampling is performed.

        Parameters
        ----------
        timesteps : Integer, optional
            The number of future timesteps to predict using the current population of particles. The default is 0.

        Returns
        -------
        future_mean : list of dictionary
            A list of prediction mean for all particles for all timesteps specified.
        future_std : list of dictionary
            A list of prediction standard deviation for all particles for all timesteps specified..

        """
        # Tracking of future predictions based on number of timesteps
        future_mean = list()
        future_std  = list()
        
        # Check if the requested number of timesteps is within range of current max range
        max_idx_range = self.index_range[1] 
        if timesteps > max_idx_range or timesteps < 0:
            print("Error: The number of timesteps exceeds the number available in reference data. No predictions made")
            return future_mean, future_std
        else:
            for i in range(timesteps-1):
                self.predict()
                future_mean.append(self.get_mean_pred())
                future_std.append(self.get_std_pred())                
                self.forward()
                
        return future_mean, future_std
#%% Support methods
    def sort_by_weight(self):
        """
        Sort particles by weight from largest to smallest. Run calculate_weights() function first before calling.

        Returns
        -------
        Numpy array-like
            Array of weights sorted from largest to smallest.

        """
        return np.argsort(-self.weights)
    
    def sort_by_score(self):
        """
        Sort particles by score.
        
        Returns
        -------
        Numpy array-like
            Array of particles sorted from largest to smallest based on score. Run calculate_score() function first before calling.

        """
        return sorted(self.particles, key=lambda x: x.score)
            
    def set_threshold(self, threshold):
        """
        Set the threshold for when particles need to be replaced. Dumb way of particle repopulation.  

        Parameters
        ----------
        threshold : float
            Sets a hard limit for when particles need to be replaced. Limit between 0 and 1. Used in conjunction with the Threshold resampling method. 

        Returns
        -------
        None.

        """
        if threshold < 1 and threshold > 0 and isinstance(threshold, numbers.Number):
            self.threshold = threshold
        else:
            print(f"WARNING: Threshold value must be a positive float value between 0 and 1. Current value is: {threshold}")
            sys.exit(1)
        
    def set_population_cut(self, population_cut):
        """
        Set the value for excluding particles from the output calculation. This is because not all particles will be well performing (especially randomly sampled particles). This number (from 0 to 1) identifies what percentage of particles to consider for calculations. 

        Parameters
        ----------
        population_cut : Float
            Percentage value between 0 and 1, determines which particles to use in calculations. 

        Returns
        -------
        None.

        """
        if population_cut < 1 and population_cut > 0 and isinstance(population_cut, numbers.Number):
            self.population_cut = population_cut
        else:        
            print(f"WARNING: Population cut value must be a positive float value between 0 and 1. Current value is: {population_cut}")
            sys.exit(1)

    def set_replacement_rate(self, replacement_rate):
        """
        Set the replacement rate for resampling particles.

        Parameters
        ----------
        replacement_rate : Float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if replacement_rate < 1 and replacement_rate >= 0 and isinstance(replacement_rate, numbers.Number):
            self.replacement_rate = replacement_rate
        else:        
            print(f"WARNING: Replacement rate value must be a positive float value between 0 and 1. Current value is: {replacement_rate}")
            sys.exit(1)
            
    def get_best_score(self):
        """
        Sort the particles by score then retrieve the best particle score. 

        Returns
        -------
        float
            Score of the best particle in the swarm.

        """
        # Reorder particles by score, higher is better
        sorted_particles = self.sort_by_score()
        return sorted_particles[0].score
    
    def get_mean_pred(self, cutoff=-1):
        """
        Calculate the mean particle predictions for one or more columns.
    
        Parameters
        ----------
        cutoff : float, optional
            If not -1, only the top fraction of particles (based on score) are used.
    
        Returns
        -------
        dict
            Dictionary of mean values for each column.
        """
        # Convert to list for uniform processing if selected_cols is a single string
        if isinstance(self.selected_cols, str):
            col = [self.selected_cols]
            
        else:
            col = self.selected_cols
            
        current_pt_mean = {key: 0.0 for key in col}        
        sorted_particles = self.sort_by_score()
        
        # Determine if only the best particles are used or all particles based on specified cutoff limit.
        if cutoff != -1:
            CF = int(len(sorted_particles)*self.population_cut)
            for pt_idx in range(CF):
                for key in col:
                    current_pt_mean[key] += sorted_particles[pt_idx].current[key]
            
            for key in col:
                current_pt_mean[key] /= CF
        
        else:
            for key in col:
                values = [p.current[key] for p in self.particles]
                current_pt_mean[key] = np.mean(values)
            
        return current_pt_mean
    
    def get_std_pred(self, cutoff=-1):
        """
        Calculate the standard deviation of particle predictions for one or more columns.
    
        Parameters
        ----------
        col : str or list of str
            Column(s) to compute the standard deviation for.
        cutoff : float, optional
            If not -1, only the top fraction of particles (based on score) are used.
    
        Returns
        -------
        dict
            Dictionary of standard deviations for each column.
        """

        # Convert to list for uniform processing
        if isinstance(self.selected_cols, str):
            col = [self.selected_cols]

        else:
            col = self.selected_cols
            
        current_pt_std = {}
        sorted_particles = self.sort_by_score()
        
        # Determine if only the best particles are used or all particles based on specified cutoff limit.
        if cutoff != -1:
            CF = int(len(sorted_particles)*cutoff)
            selected_particles = sorted_particles[:CF]
        
        else:
            selected_particles = self.particles
        
        for key in col:
            values = [pt.current[key] for pt in selected_particles]
            current_pt_std[key] = np.std(values, ddof=1)
            
        return current_pt_std        

#%% Hidden Functions    
    def __load_data__(self, tag, index):
        """
        Loads the value at the given tag and index of the particle

        Parameters
        ----------
        tag : String
            Unique identifier of the source data.
        index : Integer
            Row index within a single dataset.

        Returns
        -------
        Dictionary
            Values at current row index and document tag.

        """
        df  = self.csv_dict[tag]
        ret = df.iloc[index]
        
        return ret.to_dict()

    def __init_source_data__(self, folder_path=None, data=None, selected_cols=None):
        """
        Loads all CSV files in the specified folder into a dictionary of DataFrames.

        Parameters
        ----------
        folder_path : String, optional
            DESCRIPTION. The default is None.
        data : Dictionary, optional
            Dictionary of values to be tracked. The default is None.
        selected_cols : List of strings, optional
            Column names for either data or for each file in folder_path. The default is None.

        Returns
        -------
        None.

        """
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                print("WARNING: Input parameter data is not a pandas Dataframe, exiting")
                sys.exit(1)
            self.tags_list  = list(data.keys())
            self.csv_dict   = data
        
        elif folder_path is not None:
            self.global_path = folder_path
            
            csv_dict = {}
            
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        df             = pd.read_csv(file_path, usecols=selected_cols)
                        key            = os.path.splitext(filename)[0]
                        self.tags_list.append(key)
                        csv_dict[key]  = df
                    
                    except ValueError as e:
                                    print(f"Skipping {filename}: {e}")
    
            self.csv_dict = csv_dict
            self.global_cols = selected_cols
        
        else:
            print("Error, no data was provided to the swarm")
            self.csv_dict = None
            self.global_cols = None
            return
            
        return
    
    def __index_range_finder__(self, index_range, find_range=True):
        """
        Determine the start and end index of each data file.

        Parameters
        ----------
        index_range : Tuple
            Number of rows of data in each file. It is assumed that each file has the same number of rows. index_range = (minimum_index, maximum_index)
        find_range : Boolean, optional
            Automatically finds the range of provided data. The default is True.

        Raises
        ------
        ValueError
            Parameter index_range is not valid. The first term must be smaller than the second term; both terms must be positive

        Returns
        -------
        None.

        """
        try:
            if find_range and index_range is None:
                try:
                    # Get the minimum and maximum row index for the csv files
                    first_entry = next(iter(self.csv_dict.values()))
                    min_idx = first_entry.index.min()
                    max_idx = first_entry.index.max()
                    self.index_range = (min_idx, max_idx)
                except TypeError:
                    print("Data or folder not recognized as valid input")
                    sys.exit(0)
            
            elif not index_range is None:
                if index_range[0] > index_range[1]:
                    raise ValueError
                elif index_range[0] - index_range[1] == 0:
                    raise ValueError
                else:
                    self.index_range = index_range
            
        except ValueError:
            print("WARNING: Input parameter index_range is not a valid tuple")
        
        self.index_list = list(range(self.index_range[1] - self.index_range[0]))

    def __init_distribution__(self):
        """
        Sets a uniform probability for all keys and indices detected in the database.

        Returns
        -------
        None.

        """
        # Determine key distribution
        n = len(self.tags_list)
        if n == 0:
            print("WARNING: Dataset is empty. Assign a dataset first during initialization.")
            sys.exit(1)
            
        # Sets initial probability as uniform for the number of keys in database.
        prob = 1.0 / n
        self.tag_dist = {key: prob for key in self.tags_list}
        
        # Determine index distribution
        n = self.index_range[1] - self.index_range[0] + 1
        keys = list(range(n))
        
        # Sets initial probability as uniform for the number of indices in database
        prob = 1.0 / n
        
        self.index_dist = {key: prob for key in keys}
  