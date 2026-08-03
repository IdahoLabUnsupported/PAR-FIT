# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:45:33 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import sys
sys.path.append("../src")

import matplotlib.pyplot as plt
import torch
import numpy as np

import Function as FX
from Model_FNN import FNN
    
if __name__ == '__main__':      
    print("*********************Begin Testing*********************")       

    # Available datasets to test model
    filename = "007_Q2_015_0768_T"
    #filename = "007_Q3_024_1000_T"
    #filename = "015_Q2_016_7096_T"
    #filename = "015_Q3_016_4096_T"
    
    # Specify model name
    version   = 0o0001
    root      = './models/FNN Models/'
    modelname = 'FNN_' + filename + "_version_" + str(version)
    dataname  = 'FNN_' + filename + "_version_" + str(version) + '.csv'
    directory = root + modelname + "/"
    
    # Load Testing set
    ep_begin    = 612
    ep_end      = ep_begin+1
    interval    = 1
    
    filename = "007_Q2_015_0768_T/"
    #filename = "007_Q3_024_1000_T/"
    #filename = "015_Q2_016_7096_T/"
    #filename = "015_Q3_016_4096_T/"
    
    root = "../datasets/"
    
    path = root + filename + "/"
    time, features, dList0 = FX.load_any(ep_begin, 
                                         ep_end, 
                                         path, 
                                         interval=interval, 
                                         title='histories_short_print_')
    
    # For 007_Q2_015_0768_T and 007_Q3_024_1000_T
    TF_raw   = features[:,3].reshape(-1,1) #FL19
    UPT_raw  = features[:,8].reshape(-1,1) #TL14s1
    FCL_Temp = features[:,4].reshape(-1,1) #TA21s1
    
    dInput  = np.hstack((UPT_raw, TF_raw))
    
    # Load FNN model for testing
    engine      = torch.device("cpu")
    modelshape  = np.loadtxt(directory + modelname + "_Shape.txt", delimiter=',', dtype=int, converters=float)
    FNNmodel    = FNN(modelshape[0], modelshape[1], modelshape[2]).to(engine)
    state_dict  = torch.load(directory + modelname + ".pt")
    FNNmodel.load_state_dict(state_dict)
    FNNmodel.eval()
    
    input_test  = torch.from_numpy(dInput).to(engine)
    output_pred = (FNNmodel(input_test.float())).detach().numpy()
        
    plt.figure()
    plt.plot(time, output_pred, '--', color='green', label="FCL Prediction")
    plt.plot(time, FCL_Temp, color='black', label="Gnd Truth") 
    plt.grid()
    plt.legend()       
    plt.show()   
    print("*********************Testing Complete*********************") 