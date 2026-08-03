# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:49:18 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
# Default libraries
import sys

# Install libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# Explicit libraries
import ContourPlot as CP
import knn        
'''
Description:
    Generates the multivariate Laplacian distribution for evidence collection.

'''

class KDC:               
    def __init__(self, dSet, VERBOSE=False):   
        '''
        Constructor, must provide reference datasets. 

        Parameters
        ----------
        dSet : dArray
            Reference dataset.
            
        VERBOSE : Boolean, optional
            Provides additional software debugging information. The default is False.

        Returns
        -------
        None.

        '''
        # Setup reference dataset
        try:
            self.dSet            = pd.concat(dSet.values(), ignore_index=True).to_numpy()
            
        except:
            print("Data format is not recognized for KDC")
            sys.exit(0)
                
        # Remove duplicate samples
        u, indices           = np.unique(self.dSet, axis=0, return_index=True)
        self.dSet            = self.dSet[indices]
        
        # Optional Settings
        self.VERBOSE         = VERBOSE
        
        # Determine dimensions of input & output dimensions for covariance matrix
        data_dim             = len(self.dSet[0])
        self.dimData         = data_dim
        
        # Distribution shape parameters
        self.kernel          = 1  #specify default kernel shape 
        self.ext_r           = np.ones((1,data_dim), dtype=float)
        self.cov_all         = np.eye(data_dim, dtype=float)
        self.level           = 0.2
        self.rho             = 1
        
        # Training point search parameters
        self.radius          = -1
        self.neighN          = -1
        
    def _dist_x(self, X, S, V):
        '''
        Determines the sqrt((X-S)^T * V^-1 * (X-S)). If V is an identify matrix, then
        the return is the Euclidean distance. Otherwise, it is the Mahalanobis distance.
    
        Parameters
        ----------
        X : ndarray
            Sample point to determine probability at.
        S : ndarray
            Reference origin point of the Laplace distribution.
        V : ndarray
            Covariance matrix for input features.
    
        Returns
        -------
        ret : float
            Distance defined by, sqrt((X-S)^T * V^-1 * (X-S)).
    
        '''
        X    = X.reshape(-1,1)
        S    = S.reshape(-1,1)
        
        # if X is singular, do scalar math
        if X.flatten().size == 1: # X is singular
            X     = float(X.flatten()[0])
            S     = float(S.flatten()[0])
            V     = float(V.flatten()[0])
            V_inv = pow(V,-1)
            
            XS    = X-S
            ret   = math.sqrt(XS*V_inv*XS)
            
        # if X is matrix, do element-wise math
        else: 
            V_inv = np.copy(V)
            inv_t = np.divide(1,np.diag(V)) # Since the matrix is diagonal, it is faster to calculate the element-wise inverse
            for i in range(len(inv_t)):
                V_inv[i,i] = inv_t[i]
            XS    = X-S
            XS_T  = np.transpose(XS)
            ret   = np.sqrt(np.matmul(np.matmul(XS_T,V_inv),XS))
        
        return ret
        
    def _K_d(self, r_d):
        '''
        Calculates function value using distance from sample point using kernel
        function selected. Currently supported kernels, exponential, cosine, tophat,
        linear, and Gaussian.
    
        Parameters
        ----------
        r_d : float
            Distance to training sample calculated from dist_x().
   
        Returns
        -------
        ret : float
            Decay function value given distance.
    
        '''
        cut_func = self.kernel
        
        if cut_func==1: # "exp"
            ret    = math.exp(-2*r_d)
            
        elif cut_func==2: # "cosine"
            if r_d < 1: 
                ret    = math.cos(math.pi*r_d/2)
            else:
                ret    = 0
            
        elif cut_func==3: # "tophat"
            if r_d < 1:
                ret    = 1
            else:
                ret    = 0
                
        elif cut_func==4: # "linear"
            if r_d < 1:
                ret     = 1 - r_d
            else:
                ret     = 0
                
        elif cut_func==5: # "gaussian"
            ret    = math.exp(-math.pow(r_d,2))
        
        return ret
        
    def _Laplace_ND(self, x, s):
        '''
        Calculates the Laplace distribution for a reference point and a sample.
    
        Parameters
        ----------
        x : ndarray
            Sample point to determine probability at.
        s : ndarray
            Reference origin point of the Laplace distribution.
        sigma : ndarray
            Covariance matrix for input features.
        alpha : float, optional
             Arbitrary scaling factor. The default is 2.
    
        Returns
        -------
        ret : float
            The probability of 'x' given Laplace distribution origin at 's'.
    
        '''    
        V       = self.cov_all
        r_d     = self._dist_x(x,s,V)
        ret     = self._K_d(r_d)
        
        return ret
    
    def evaluate_points(self, dInput):
        dInput = np.array(list(dInput.values()))
        
        # Find closest points to dInput
        d_all,index_x,nn_x = knn.k_nearest(dInput, self.dSet, n_neighbors=self.neighN, cut_off_r=self.radius)
        
        # Determine confidence based on single point for all training samples
        prob_list  = list()
        
        for index in range(len(d_all)):
            if d_all[index] == -1:
                prob_list.append(0)
            else:
                train_point  = nn_x[index]
                prob         = self._Laplace_ND(dInput,train_point)   
                prob_list.append(prob)
        
        prob_list      = np.vstack(prob_list)
        P_x_all        = np.sum(prob_list)
        
        # Determine reliability of point relative to density of points around it
        P_x = P_x_all/self.rho ##
        #P_x = np.amax(prob_list) ##
        
        # Determine reliability of point relative to max function
        
        if P_x > 1: ##
            P_x = 1 ##
            
        return P_x
    
    def _optimize_cov(self, ext_rd):
        '''
        Generates optimal covariance function given kernel function and desired
        extrapolation radius from points.

        Parameters
        ----------
        ext_rd : ndarray, float
            An 1-D array of float values of the desired radii in all axeses.

        Returns
        -------
        V_new : ndarray, float
            2D array with optimal covariance values for proper kernel structure.

        '''
        lvl    = self.level
        V_old  = self.cov_all
        V_diag = np.diag(V_old)
        V_new  = np.copy(V_old)
        
        cut_func = self.kernel
        
        if cut_func==1: # "exp"
            for i in range(len(V_diag)):
                x_r        = ext_rd[i]
                v_ii_new   = pow((math.log(lvl)/2),-2) * pow(x_r,2)
                V_new[i,i] = v_ii_new
            
        elif cut_func==2: # "cosine"
            for i in range(len(V_diag)):
                x_r        = ext_rd[i]
                temp       = 2 * math.acos(lvl) / math.pi
                v_ii_new   = pow(x_r,2) * pow(temp,-2)
                V_new[i,i] = v_ii_new                
            
        elif cut_func==3: # "tophat"
            for i in range(len(V_diag)):
                x_r        = ext_rd[i]
                v_ii_new   = math.pow(x_r,2)
                V_new[i,i] = v_ii_new   
                
        elif cut_func==4: # "linear"
            for i in range(len(V_diag)):
                x_r        = ext_rd[i]
                v_ii_new   = math.pow(x_r,2) / math.pow(1-lvl,2)
                V_new[i,i] = v_ii_new 
                
        elif cut_func==5: # "gaussian"
            for i in range(len(V_diag)):
                x_r        = ext_rd[i]
                v_ii_new   = math.pow(x_r,2) / (-math.log(lvl))
                V_new[i,i] = v_ii_new     
        
        self.cov_all   = V_new
        self.ext_r     = ext_rd
        
        return 
        
    def mod_kernel(self, k):
        '''
        Modify distribution kernel. 

        Parameters
        ----------
        k : kernel option
            k=1 Exponential
            k=2 Cosine
            k=3 Tophat
            k=4 Linear
            k=5 Gaussian.

        Returns
        -------
        int
            Status of operation.

        '''
        if k!=1 and k!=2 and k!=3 and k!=4 and k!=5:
            print("Invalid kernel function, no modification made.")
            return 0
        self.kernel=k
        return 1        
    
    def mod_ext_radius(self, ext_rd):
        '''
        Modify extrapolation radius of each feature axis.

        Parameters
        ----------
        ext_rd : ndarray, float
            1-D array of user-specified extrapolation distance for each feature.

        Returns
        -------
        int
            Status of operation.

        '''
        if ext_rd.any() < 0:
            print("Negative distance not allowed, no modification made.")
            return 0   
        
        self.ext_r  = ext_rd.reshape(1,-1).astype(np.float32)  #Flatten
        self._optimize_cov(ext_rd) #Change covariance matrix to adjust for new ext. distance.
        return 1
    
    def mod_neighbors(self, nn):
        if nn <= 0 and nn != -1:
            print("Invalid nearest neighbor rule, no modification made.")
            return 0            
        
        self.neighN  = nn
        self.rho     = nn
        return 1



