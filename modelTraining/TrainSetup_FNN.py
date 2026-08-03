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
import numpy as np

from Model_FNN import FNN

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
          learning_rate = 0.01, 
          weight_decay = 1E-5,
          output_dim=1, 
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
    
    :param learning_rate: Learning rate for machine learning model. The default is 0.01
    :type learning_rate: float, optional
    
    :return ideal: The best model from training.
    :rtype ideal: nn.model
    
    :return min_log_loss: The minimum log loss from the best trained model
    :rtype min_log_loss: float

    """
    engine = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_data, output_data = src_data_kf.create_dataset(batch_size)
    
    #Dimenions of neural network
    input_dim = len(input_data[0,:])
    hidden_dim = hidden_dim
    output_dim = len(output_data[0,:])
    
    #Define model type
    model = FNN(input_dim, hidden_dim, output_dim)
    
    #Define loss metric, optimizer, and runtime engine
    loss_fn = torch.nn.MSELoss()   
    wd_lamda = weight_decay
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd_lamda, lr=learning_rate)
    model = model.to(engine)
    
    num_epochs = num_epochs
    min_log_loss = 1000000;
    ideal = None;
    flag = 0;

    for epoch in range(num_epochs):        
            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            input_training = torch.from_numpy(input_data).to(engine)
            output_training = torch.from_numpy(output_data).to(engine)
        
            y_pred = model(input_training.float())
    
            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            epoch_loss = custom_loss_fn(y_pred, output_training.float(), loss_fn)
                
            if epoch_loss.item() < min_log_loss:
                if abs(epoch_loss.item() - min_log_loss) > min_loss:
                    flag = 0
                
                min_log_loss = epoch_loss.item()
                ideal = FNN(input_dim, hidden_dim, output_dim)
                    
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
            
            flag = flag + 1
            
            # Recreate the batch training set
            input_data, output_data = src_data_kf.create_dataset(batch_size)
            
    return ideal, min_log_loss       
    
def custom_loss_fn(pred, target, loss_fn=None):
    """
    Use a different loss function to train the network. Unused.
    
    :param pred: Input samples.
    :type pred: ndarray
    
    :param target: Ground truth.
    :type target: ndarray
    
    :return loss: The loss calculated based on the loss function
    :rtype loss: ndarray
    """
    loss = loss_fn(pred,target)
    return loss

def verify_model_mse(src_data_kf, model):
    """
    Verifies that model mse from training is the true model mse

    :param src_data_kf: Source data to use
    :type src_data_kf: Data_kf
    
    :param model: Pre-trained model
    :type modeL nn.model

    :return mse.item(): The mean square loss from the model predictions
    :rtype mse.item(): float

    """
    engine = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_data, output_data = src_data_kf.create_dataset()   
    
    loss_fn = torch.nn.MSELoss()
    model = model.to(engine)

    input_test = torch.from_numpy(input_data).to(engine)
    output_test = torch.from_numpy(output_data).to(engine)
    output_pred_test0 = model(input_test.float())
    mse = loss_fn(output_pred_test0, output_test.float())
    
    return mse.item()