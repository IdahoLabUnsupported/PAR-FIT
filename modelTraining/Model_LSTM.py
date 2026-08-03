# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import torch
    
class LSTM(torch.nn.Module):
    """
    Creates a 4 layer feedforward NN, 1 input, 1 output, 2 hidden. Each layer is separated by a ReLU layer.
    
    :param input_dim: Size of the input layer dimension
    :type input_dim: int
    
    :param hidden_dim: Size of the hidden layer input dimension
    :type hidden_dim: int
    
    :param output_dim: Size of the output layer dimension
    :type output_dim: int
    """
    def __init__(self, n_features, seq_length, n_hidden, n_layers, n_output, dropout_p = 0):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len    = seq_length
        self.n_hidden   = n_hidden
        self.n_layers   = n_layers
        self.n_output   = n_output

        self.l_lstm     = torch.nn.LSTM(input_size = n_features, 
                                        hidden_size = self.n_hidden,
                                        num_layers = self.n_layers, 
                                        batch_first = True)
        
        self.dropout    = torch.nn.Dropout(p=dropout_p)
        self.l_linear   = torch.nn.Linear(self.n_hidden*self.seq_len, self.n_output)
        
        return

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state   = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden  = (hidden_state, cell_state)

        return

    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        lstm_out = self.dropout(lstm_out) 
        x = lstm_out.contiguous().view(batch_size,-1)
        
        return self.l_linear(x)