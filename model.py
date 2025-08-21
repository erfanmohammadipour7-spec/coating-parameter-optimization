# model.py

import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for regression.
    """
    def __init__(self, input_size, output_size, hidden_neurons=16, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            # Input layer and first hidden layer
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(hidden_neurons // 2, output_size)
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.layers(x)
