# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import sys
sys.path.append("../legacy")

import os
from datetime import date

import matplotlib.pyplot as plt
import math
import torch
import numpy as np

import Function as FX
import Data_kf as dKF
import TrainSetup_FNN as mDT
from Model_FNN import FNN
    
if __name__ == '__main__' and True:   
    # Load Training set
    ep_begin = 256
    ep_end   = 500
    interval = 2
    
    # Available datasets to train model
    filename = "007_Q2_015_0768_T"
    #filename = "007_Q3_024_1000_T"
    #filename = "015_Q2_016_7096_T"
    #filename = "015_Q3_016_4096_T"
    
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
    dSet    = np.hstack((dInput, FCL_Temp))
    np.random.shuffle(dSet)
    
    # Cut training set to be fraction of total
    cut     = math.floor(len(dSet)*1)
    dTrain  = dSet[:cut,:2]
    dTarget = dSet[:cut,2].reshape(-1,1)
    saveSet = dSet[:cut,:]

    # Hyperparameters
    input_dim      = len(dInput[0])     # this is number of parallel inputs
    output_dim     = len(dTarget[0])    # this is number of outputs
    
    n_hidden       = 30                 # this is number of hidden states
    n_layers       = 3                  # this is number of hidden layers
    l_rate         = 0.001              # this is learning rate
    train_epoch    = 10000              # this is epoch number
    batch_size     = 64                 # this is batch size
    weight_decay   = 0.05*batch_size/len(dTrain) # this is regularization weight
    
    target_loss    = 1e-10              # this is the target training loss
    
    # Save file names
    version   = 0o0001
    root      = './models/FNN Models/'
    modelname = 'FNN_' + filename + "_version_" + str(version)
    dataname  = 'FNN_' + filename + "_version_" + str(version) + '.csv'
    directory = root + modelname + "/"
    new_model = False
    
    print(f"Model location: {directory}\n")
    q = input("Confirm train new model? (y/n)")
    if (q=='Y' or q=='y'):       
        new_model = True
            
    if new_model == True:
        print("*********************Begin Training*********************")
          
        kf = dKF.Data_kf(dTrain, dTarget)
        best_model, mse = mDT.train(kf, 
                                    batch_size=batch_size, 
                                    learning_rate=l_rate, 
                                    hidden_dim=n_hidden,
                                    min_loss=target_loss, 
                                    num_epochs=train_epoch,
                                    weight_decay=weight_decay,
                                    VERBOSE=True)
        
        # Setup folder for model settings
        os.makedirs(directory, exist_ok=True)

        # Save model and parameter configurations
        torch.save(best_model.state_dict(), directory+modelname+".pt")
        
        print("Ideal model mse " + str(mse))
        
        # Save model development information        
        save_txt   = list()
        save_txt.append("******************MODEL CONFIGURATION******************"+ "\n")
        save_txt.append("Date of Creation: " + str(date.today())+ "\n")
        save_txt.append("Model name: " + modelname + "\n"+ "\n")
        
        save_txt.append("Input dimension: " + str(input_dim) + "\n")
        save_txt.append("Hidden layers: " + str(n_layers) + "\n")
        save_txt.append("Nodes per hidden layer: " + str(n_hidden) + "\n")
        save_txt.append("Output dimension: " + str(output_dim) + "\n"+ "\n")
        
        save_txt.append("******************TRAINING CONFIGURATION******************"+ "\n")
        save_txt.append("Number of epochs: " + str(train_epoch)+ "\n")
        save_txt.append("Learning rate: " + str(l_rate)+ "\n")
        save_txt.append("Loss function: MSE"+ "\n")
        save_txt.append("Training set size: " + str(int((ep_end-ep_begin)/interval))+ "\n")
        save_txt.append("Training set origin: " + dataname + "\n")
        save_txt.append("Batch size: " + str(batch_size) + "\n")
        save_txt.append("Stop loss: " + str(target_loss)+ "\n")
        save_txt.append("Weight decay: " + str(weight_decay) + "\n")
        save_txt.append("Model post training error: " + str(mse) + "\n")
        save_txt.append("Training set start: "+ str(ep_begin)+ "\n")
        save_txt.append("Training set end: "+ str(ep_end)+ "\n")
        
        # Save dSet for later use in evaluation of LD_conf
        arr_sav = np.hstack((dTrain, dTarget))
        np.savetxt(directory+dataname, saveSet, delimiter=",")
            
        # Write data to text file
        with open(directory + modelname + '_Specifications.txt', 'w') as f:
            f.writelines(save_txt)
        f.close()
        
        # Save model shape 
        save_shape = np.array([input_dim, n_hidden, output_dim]).reshape(1,-1)
        np.savetxt(directory + modelname + "_Shape.txt", save_shape, delimiter=',')

        print("*********************Training Complete*********************")
    
    else:
        print("Exiting, no model trained")