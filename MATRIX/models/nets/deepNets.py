import torch
import torch.nn as nn

class DeepSurvFFNN(nn.Module):

    """
    Neural network architecture for DeepSurv.
    """
    
    def __init__(self, num_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):    
        super(DeepSurvFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = num_inputs
        for hidden_size in (hidden_layers or []):
            # Dense layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            self.layers.append(activation_fn())
            
            # Dropout
            if dropout > 0.0:
                self.layers.append(nn.Dropout(p=dropout))
            
            input_size = hidden_size
        
        # Output layer (log hazard ratio)
        self.output_layer = nn.Linear(input_size, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.output_layer(x)

class DeepMultiTaskFFNN(nn.Module):

    """
    Neural network architecture for DeepMultiTask.
    """
    
    def __init__(self, num_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepMultiTaskFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = num_inputs
        for hidden_size in (hidden_layers or []):
            # Dense layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            self.layers.append(activation_fn())
            
            # Dropout
            if dropout > 0.0:
                self.layers.append(nn.Dropout(p=dropout))
            
            input_size = hidden_size
        
        # Output layer (log hazard ratio)
        self.cox_output = nn.Linear(input_size, 4)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        cox_output = self.cox_output(x)

        return cox_output
    
class DeepMultiTaskMultiLossFFNN(nn.Module):

    """
    Neural network architecture for DeepMultiTaskMultiLoss.
    """
    
    def __init__(self, num_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepMultiTaskMultiLossFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = num_inputs
        for hidden_size in (hidden_layers or []):
            # Dense layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            self.layers.append(activation_fn())
            
            # Dropout
            if dropout > 0.0:
                self.layers.append(nn.Dropout(p=dropout))
            
            input_size = hidden_size
        
        # Output layer (log hazard ratio)
        self.cox_output = nn.Linear(input_size, 4)
        # Output layer (binary classification)
        self.binary_output = nn.Linear(input_size, 4)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        cox_output = self.cox_output(x)
        binary_output = self.binary_output(x)

        return torch.cat((cox_output, binary_output), dim=1)