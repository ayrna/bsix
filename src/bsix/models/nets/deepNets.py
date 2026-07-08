import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(activation):

    """
    Resolve an activation-function name to its nn.Module class.
    """
    try:
        return _ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(f"Unknown activation function: {activation}")


def _build_hidden_layers(input_size, hidden_layers, activation_fn, dropout=0.0, batch_norm=False):

    """
    Build a list of (Linear -> [BatchNorm] -> Activation -> [Dropout]) blocks.
    """
    layers = []
    for hidden_size in (hidden_layers or []):
        # Fully connected layer
        layers.append(nn.Linear(input_size, hidden_size))

        if batch_norm:
            # Batch normalization layer
            layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(activation_fn())

        if dropout > 0.0:
            # Dropout layer
            layers.append(nn.Dropout(p=dropout))

        input_size = hidden_size

    output_size = input_size

    return layers, output_size


class DeepSurvFFNN(nn.Module):

    """
    Neural network architecture for DeepSurv.
    """

    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepSurvFFNN, self).__init__()

        activation_fn = _get_activation(activation)
        layers, output_size = _build_hidden_layers(number_inputs, hidden_layers, activation_fn, dropout, batch_norm)
        self.layers = nn.ModuleList(layers)

        # Output layer (log hazard ratio)
        self.output_layer = nn.Linear(output_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.output_layer(x)


class DeepMultiTaskFFNN(nn.Module):

    """
    Neural network architecture for DeepMultiTask.
    """

    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False,
                 number_outputs=4):
        super(DeepMultiTaskFFNN, self).__init__()

        activation_fn = _get_activation(activation)
        layers, output_size = _build_hidden_layers(number_inputs, hidden_layers, activation_fn, dropout, batch_norm)
        self.layers = nn.ModuleList(layers)

        # Output layer (log hazard ratio per task)
        self.cox_output = nn.Linear(output_size, number_outputs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.cox_output(x)


class DeepMultiTaskMultiLossFFNN(nn.Module):

    """
    Neural network architecture for DeepMultiTaskMultiLoss.
    """

    def __init__(self, number_inputs, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False, number_cox_outputs=4, number_binary_outputs=4):
        super(DeepMultiTaskMultiLossFFNN, self).__init__()

        activation_fn = _get_activation(activation)
        layers, output_size = _build_hidden_layers(number_inputs, hidden_layers, activation_fn, dropout, batch_norm)
        self.layers = nn.ModuleList(layers)

        # Output layer (log hazard ratio)
        self.cox_output = nn.Linear(output_size, number_cox_outputs)
        # Output layer (binary classification)
        self.binary_output = nn.Linear(output_size, number_binary_outputs)

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

    def __init__(self, number_inputs, number_events, number_categories, hidden_layers_shared=None, hidden_layers_specific=None, activation="relu", dropout=0.0, batch_norm=False):
        super(DeepHitFFNN, self).__init__()
        self.number_events = number_events
        self.number_categories = number_categories

        activation_fn = _get_activation(activation)

        # Build shared sub-network
        shared_layers, shared_output_size = _build_hidden_layers(number_inputs, hidden_layers_shared, activation_fn, dropout, batch_norm)
        self.shared_net = nn.Sequential(*shared_layers)

        # Output shared sub-network dimension
        self.shared_output_dimension = shared_output_size if (hidden_layers_shared and len(hidden_layers_shared) > 0) else 0

        # Residual connection: each specific sub-network sees the raw inputs concatenated with the shared sub-network's output.
        specific_input_size = number_inputs + self.shared_output_dimension

        # Build one specific sub-network per progression
        self.specific_nets = nn.ModuleList()
        for _ in range(self.number_events):
            specific_layers, specific_output_size = _build_hidden_layers(specific_input_size, hidden_layers_specific, activation_fn, dropout, batch_norm)
            self.specific_nets.append(nn.Sequential(*specific_layers))

        # Output specific sub-networks dimension
        self.specific_output_dimension = specific_output_size

        # Final output layer
        self.output_layer = nn.Linear(self.specific_output_dimension * self.number_events, self.number_categories * self.number_events)

    def forward(self, x):
        shared_output = self.shared_net(x)
        specific_input = torch.cat([x, shared_output], dim=1) if self.shared_output_dimension > 0 else x

        specific_outputs = [net(specific_input) for net in self.specific_nets]
        specific_outputs = torch.stack(specific_outputs, dim=1)
        specific_outputs = specific_outputs.reshape(specific_outputs.shape[0], -1)

        output = self.output_layer(specific_outputs)

        output = torch.softmax(output, dim=1)
        output = output.reshape(-1, self.number_events, self.number_categories)

        return output