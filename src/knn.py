# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import numpy as np
import math

def distance_euclidean(A, B):
    '''
    Calculates the euclidean distance between two points.

    Parameters
    ----------
    A : ndarray
        Point vector.
    B : ndarray
        Point vector.

    Returns
    -------
    RET: float
        Distance between two points.

    '''
    # Calculates Euclidean distance between 2 vectors
    return math.dist(A,B)
    
def k_nearest(single, t_set, n_neighbors=-1, cut_off_r=-1):
    '''
    Obtains the 5 nearest neighbors to a single point in the given training set

    Parameters
    ----------
    single : ndarray
        A 1-D vector of features that describe the single point.
    t_set : ndarray
        A list of 1-D vectors of the training samples used for the NN.

    Returns
    -------
    ret: ndarray
        An ndarray of the 'n_neighbors' closest samples from the t_set.

    '''
    dist_list = list()
    A         = single
    index     = 0
    
    for B in t_set:
        dist_AB = distance_euclidean(A, B)
        dist_list.append([dist_AB, index, B])
        index += 1
    
    sort_list   = sorted(dist_list, key=lambda x: x[0])
    temp        = np.array(sort_list, dtype=object)
    
    # if temp has a single element, don't sort.
    if len(temp) == 1:
        sort_list = temp
    else:
        sort_list   = np.array(sort_list[0:n_neighbors],dtype=object)
    
    if cut_off_r != -1 and cut_off_r > 0:
        for i in range(len(sort_list)):
            row = sort_list[i]
            if row[0] <= cut_off_r:
                continue
            else:
                sort_list = sort_list[:i]
                break
    
    try:
        dist_arr  = np.vstack(sort_list[:,0])
        index_arr = np.vstack(sort_list[:,1])
        coor_arr  = np.vstack(sort_list[:,2])
    
    except ValueError:
        dist_arr  = np.array([-1])
        index_arr = np.array([])
        coor_arr  = np.array([])
        
    return dist_arr, index_arr, coor_arr

    