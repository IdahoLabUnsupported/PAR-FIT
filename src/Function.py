# -*- coding: utf-8 -*-
"""
Last modified: Mar. 4, 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import numpy as np

def subsample(dSet, n=23):
    '''
    Visualization of reference data reliability maps do not require all data points.
    This function subsamples the data to generate more visually appeasing figures.

    Parameters
    ----------
    dSet : ndarray
        Data array to subsample down.
    n : int
        Subsampling interval. Default n is 23.
        
    Returns
    -------
    dNew : ndarray
        New data array with subsamples.

    '''
    cntr = 0
    new_a= list()
    for i in dSet:
        # checking if element is nth pos
        if(cntr % n == 0):
            new_a.append(i.reshape(1,-1))
        # incrementing counter
        cntr += 1
    
    # Convert to ndarray
    dNew   = np.vstack(new_a)
    
    return dNew

def load_any(ep_begin, ep_end, path, interval=1, title='', start=0, end=-1):
    '''
    Load all unformated uncompressed data

    Parameters
    ----------
    ep_begin : int
        Beginning episode.
    ep_end : int
        Last episode.
    path : path
        Path to data file object.
    interval : int, optional
        Skipping value. The default is 1.
    start : int, optional
        If dataset contains multiple entries, specifies the starting index. The default is 0.
    end : int, optional
        If dataset contains multiple entries, specifies the ending index. The default is -1 or all entires.

    Returns
    -------
    raw_input_test0 : ndArray
        Model input as array.
    raw_output_test0 : ndArray
        Intended model output as array.
    time : ndArray
        Vector of time stamps.
    dList0 : ndArray
        Metadata; contains all information.
    '''
    dList0 = []

    #Get data
    for n in range(ep_begin, ep_end):
        if n%interval==0:
            file_path = path + title + str(n) + ".csv"
            temp      = np.genfromtxt(file_path, delimiter=',', skip_header=True) 
            dList0.append(temp[start:end, :])
            
    all_list = np.vstack(dList0)
    
    time     = all_list[:,0]
    features = all_list[:,0:]    
    
    return time, features, dList0