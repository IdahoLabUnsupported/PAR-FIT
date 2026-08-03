# -*- coding: utf-8 -*-
"""
Last modified: July 12, 2024
    
Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import torch.nn as nn

# Define Neural network for diagnosis
class FNN(nn.Module):
    """
    Creates a 4 layer feedforward NN, 1 input, 1 output, 2 hidden. Each layer is separated by a ReLU layer.
    
    :param input_dim: Size of the input layer dimension
    :type input_dim: int
    
    :param hidden_dim: Size of the hidden layer input dimension
    :type hidden_dim: int
    
    :param output_dim: Size of the output layer dimension
    :type output_dim: int
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Constructor
        super(FNN, self).__init__()
        
        # Linear function 1, 4:20 (Scaling)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        # Linear function 2, 20:20
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        # Linear function 3, 20:20 
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 20 --> 1
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        # Network dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward propagate a new sample input through the network.
        
        :param x: Input parameter with size input_dim
        :type x: np.array
        
        :return out: Array of output of size output_dim
        :rtype: np.array

        """
        # Layer 1
        out = self.fc1(x)
        out = self.relu1(out)

        # Layer 2
        out = self.fc2(out)
        out = self.relu2(out)

        # Layer 3
        out = self.fc3(out)
        out = self.relu3(out)
    
        # Sigmoid function 4 (readout)
        out = self.fc4(out)

        return out