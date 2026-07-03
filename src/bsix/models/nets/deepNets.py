import torch
import torch.nn as nn

class DeepSurvFFNN(nn.Module):

    """
    Neural network architecture for DeepSurv.
    """
    
    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):    
        super(DeepSurvFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "elu":
            activation_fn = nn.ELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = number_inputs
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
    
    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepMultiTaskFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "elu":
            activation_fn = nn.ELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = number_inputs
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
    
    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepMultiTaskMultiLossFFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "elu":
            activation_fn = nn.ELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build hidden layers
        input_size = number_inputs
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
    
class DeepHitFFNN(nn.Module):

    """
    Neural network architecture for DeepHit.
    """

    def __init__(self, number_inputs, number_progressions, number_events, number_categories, hidden_layers_shared=None, hidden_layers_specific=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepHitFFNN, self).__init__()
        self.number_progressions = number_progressions
        self.number_events = number_events
        self.number_categories = number_categories

        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        elif activation == "elu":
            activation_fn = nn.ELU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Build shared sub-network
        shared_layers = []
        input_size = number_inputs
        
        for hidden_size in (hidden_layers_shared or []):
            # Dense layer
            shared_layers.append(nn.Linear(input_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            shared_layers.append(activation_fn())
            
            # Dropout
            if dropout > 0.0:
                shared_layers.append(nn.Dropout(p=dropout))
            
            input_size = hidden_size
        
        # Packaging shared-subnetwork
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Output shared sub-network dimension
        self.shared_output_dimension = input_size if (hidden_layers_shared and len(hidden_layers_shared) > 0) else 0

        # Residual connection
        specific_input_size = number_inputs + self.shared_output_dimension

        # Build specific sub-networks
        self.specific_nets = nn.ModuleList()
        
        for _ in range(self.number_progressions):
            specific_layers = []
            input_size = specific_input_size
            
            # Build each specific sub-network
            for hidden_size in (hidden_layers_specific or []):
                # Dense layer
                specific_layers.append(nn.Linear(input_size, hidden_size))
                
                # Batch normalization
                if batch_norm:
                    specific_layers.append(nn.BatchNorm1d(hidden_size))
                
                # Activation
                specific_layers.append(activation_fn())
                
                # Dropout
                if dropout > 0.0:
                    specific_layers.append(nn.Dropout(p=dropout))
                
                input_size = hidden_size
                
            # Packaging specific-subnetworks
            self.specific_nets.append(nn.Sequential(*specific_layers))

        # Output specific sub-networks dimension
        self.specific_output_dimension = input_size

        # Final output layer
        self.output_layer = nn.Linear(self.specific_output_dimension * self.number_progressions * self.number_events, self.number_categories * self.number_progressions)

        self._init_weights()

    def _init_weights(self):

        """
        Xavier initialisation, used as a baseline.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared_output = self.shared_net(x)
        specific_input = torch.cat([x, shared_output], dim=1) if self.shared_output_dimension > 0 else x

        specific_outputs = [net(specific_input) for net in self.specific_nets]
        specific_outputs = torch.stack(specific_outputs, dim=1)
        specific_outputs = specific_outputs.reshape(specific_outputs.shape[0], -1)

        output = self.output_layer(specific_outputs)
        output = output.reshape(-1, self.number_progressions, self.number_events * self.number_categories)
        output = torch.softmax(output, dim=2)
        output = output.reshape(-1, self.number_progressions, self.number_events, self.number_categories)

        return output