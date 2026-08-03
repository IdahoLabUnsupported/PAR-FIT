# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
import sys
sys.path.append("../legacy")

import copy
import torch
from torch.optim import lr_scheduler
import numpy as np

import Model_LSTM

def split_sequences(dSet, n_steps):
    X = list()
    Y = list()
    
    for i in range(len(dSet)-n_steps):
        end_ix = i+n_steps
        if end_ix > len(dSet):
            break
        seq_x = dSet[i:end_ix, :-1]
        seq_y = dSet[end_ix, -1]
        X.append(seq_x)
        Y.append(seq_y)
        
    return np.array(X), np.array(Y).reshape(-1,1)

def train(src_data_kf, 
          batch_size=-1, 
          num_epochs=150000, 
          min_loss=2.5, 
          hidden_dim=20, 
          learning_rate=0.01,
          layers=4,
          seq_length=1,
          patience=40,
          timesteps=1,
          weight_decay=1e-5,
          VERBOSE=False):
    """
    Trains the model based on the input parameters given and source data

    :param src_data_kf: Source data
    :type src_data_kf: Data_kf   
    
    :param num_epochs: Number of training epochs. The default is 150000.
    :type num_epochs: int, optional
    
    :param min_loss: Maximum acceptable mean square error loss. The default is 2.5.
    :type min_loss: float64, optional
    
    :param hidden_dim: Number of nodes per hidden layer. The default is 20.
    :type hidden_dim:  int, optional
    
    :param output_dim: Number of nodes per output layer. The default is 1.
    :type output_dim: int, optional
    
    :return ideal: The best model from training.
    :rtype ideal: nn.model
    
    :return min_log_loss: The minimum log loss from the best trained model
    :rtype min_log_loss: float

    """
    engine = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_data, output_data = src_data_kf.create_dataset_nonRandom(batch_size)
    
    #Dimenions of neural network
    input_dim = len(input_data[0,:])
    hidden_dim = hidden_dim
    output_dim = len(output_data[0,:])
    
    #Define model type
    model = Model_LSTM.LSTM(n_features=input_dim,
                            seq_length=timesteps,
                            n_hidden=hidden_dim,
                            n_layers=layers,
                            n_output=output_dim,
                            dropout_p=0)

    model = model.to(engine)
    #criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss(reduction='mean') # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = 0.5, min_lr = 1e-9, verbose = False)

    #Define loss metric, optimizer, and runtime engine   
    num_epochs = num_epochs
    min_log_loss = 10000;
    ideal = None;
    flag = 0;

    for epoch in range(num_epochs):        
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            model.train()
            dSet = np.hstack((input_data, output_data))
            X, y = split_sequences(dSet, timesteps)
            
            x_batch = torch.tensor(X,dtype=torch.float32).to(engine)
            y_batch = torch.tensor(y,dtype=torch.float32).to(engine)

            model.init_hidden(x_batch.size(0))
            output = model(x_batch)
            epoch_loss = criterion(output, y_batch)
                
            if epoch_loss.item() < min_log_loss:
                if abs(epoch_loss.item() - min_log_loss) > 0.0001:
                    flag = 0
                
                min_log_loss = epoch_loss.item()
                print(min_log_loss)
                ideal = Model_LSTM.LSTM(input_dim, timesteps, hidden_dim, layers, output_dim)
                ideal.load_state_dict(copy.deepcopy(model.state_dict()))
            
            if epoch_loss.item() < min_loss or flag == 75000:
                break
            
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()
    
            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            epoch_loss.backward()
            optimizer.step()  
            
            # Every 500 epochs, print the status of training.
            if epoch%500 == 0:
                print("Epoch : " + str(epoch) + "/" + str(num_epochs))
                if VERBOSE:
                    print(min_log_loss)
            
            # Recreate the batch training set
            input_data, output_data = src_data_kf.create_dataset_nonRandom(batch_size)
            
            flag = flag + 1
            
    return ideal, min_log_loss