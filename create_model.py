import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(in_features, nodes, out_features):
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            # Create a list of layers for the hidden part of the network
            self.layers = nn.ModuleList()
            current_in = in_features
            for node in nodes:
                self.layers.append(nn.Linear(current_in, node))
                current_in = node
            
            # Define the output layer
            self.out = nn.Linear(current_in, out_features)
        
        def forward(self, x):
            # Pass the input through each hidden layer with a ReLU activation
            for layer in self.layers:
                x = F.relu(layer(x))
            # Pass through the output layer (activation can be added if needed)
            x = self.out(x)
            return x

    return CustomModel()