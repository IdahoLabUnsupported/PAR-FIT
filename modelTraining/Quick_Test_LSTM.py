# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:54:03 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import sys
sys.path.append("../legacy")

import matplotlib.pyplot as plt
import torch
import numpy as np

import Function as FX
import TrainSetup_LSTM as mDT
from Model_LSTM import LSTM
    
if __name__ == '__main__':    
    print("*********************Begin Testing*********************")       
    # Available datasets to test model
    #filename = "007_Q2_015_0768_T"
    #filename = "007_Q3_024_1000_T"
    #filename = "015_Q2_016_7096_T"
    filename = "015_Q3_016_4096_T"
    
    # Load model 
    version   = 0o0004
    root      = './models/LSTM Models/'
    modelname = 'LSTM_' + filename + "_version_" + str(version)
    dataname  = 'LSTM_' + filename + "_version_" + str(version) + '.csv'
    directory = root + modelname + "/"
    
    # Load Testing set
    range_t     = 2002  # Number of time steps
    ep_begin    = 2
    ep_end      = ep_begin+1
    interval    = 1
    
    root = "../datasets/"
    
    path = root + filename + "/"
    time, features, dList0 = FX.load_any(ep_begin, 
                                         ep_end, 
                                         path, 
                                         interval=interval, 
                                         title='histories_',
                                         start=0,
                                         end=1000)
    
    # For 007_Q2_015_0768_T and 007_Q3_024_1000_T
    TF_raw   = features[:,3].reshape(-1,1) #FL19
    UPT_raw  = features[:,8].reshape(-1,1) #TL14s1
    FCL_Temp = features[:,4].reshape(-1,1) #TA21s1
    
    dInput  = np.hstack((UPT_raw, TF_raw))
    dSet    = np.hstack((dInput, FCL_Temp))
    
    # Load FNN model for testing
    engine      = torch.device("cpu")
    modelshape  = np.loadtxt(directory + modelname + "_Shape.txt", delimiter=',', dtype=int)
    X, y    = mDT.split_sequences(dSet, int(modelshape[3]))
                                  
    LSTMmodel   = LSTM(n_features=modelshape[0],
                       n_hidden=int(modelshape[1]),
                       n_output=int(modelshape[2]),
                       seq_length=int(modelshape[3]),
                       n_layers=int(modelshape[4])).to(engine)
    
    state_dict  = torch.load(directory + modelname + ".pt")
    LSTMmodel.load_state_dict(state_dict)
    LSTMmodel.eval()
    
    input_test  = torch.from_numpy(X).to(engine)
    LSTMmodel.init_hidden(input_test.size(0))
    output_pred = (LSTMmodel(input_test.float())).detach().numpy()
        
    plt.figure()
    plt.plot(time[:len(time)-int(modelshape[3])], output_pred, '--', color='green', label="FCL Prediction")
    plt.plot(time, FCL_Temp, color='black', label="Gnd Truth") 
    plt.grid()
    plt.legend()       
    plt.show()
