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

    Parameters
    ----------
    activation : str
        Name of the activation function (e.g., "relu", "selu", "elu", "tanh", "sigmoid").

    Returns
    -------
    type
        The corresponding `torch.nn` module class for the activation.
    """

    try:
        return _ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(f"Unknown activation function: {activation}")


def _build_hidden_layers(input_size, hidden_layers, activation_fn, dropout=0.0, batch_norm=False):

    """
    Build a list of (Linear -> [BatchNorm] -> Activation -> [Dropout]) blocks.

    Parameters
    ----------
    input_size : int
        Size of the input feature vector for the first hidden layer.
    hidden_layers : list or None
        Sequence of integers with sizes for each hidden layer. If ``None`` or
        empty then no hidden layers are created.
    activation_fn : callable
        A callable returning an activation `nn.Module` (e.g., class from `_ACTIVATIONS`).
    dropout : float, optional
        Dropout probability to apply after each activation (default: 0.0).
    batch_norm : bool, optional
        If True, insert a `nn.BatchNorm1d` layer after each linear layer.

    Returns
    -------
    tuple
        A tuple `(layers, output_size)` where `layers` is a list of
        `torch.nn` modules composing the hidden blocks and `output_size` is
        an integer with the size of the last hidden layer (or the original
        `input_size` if no hidden layers are provided).
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

        """
        Initialize the DeepSurv feed-forward neural network.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        hidden_layers : list or None, optional
            Sizes of hidden layers (default: None).
        activation : str, optional
            Activation function name to use for hidden layers (default: "relu").
        dropout : float, optional
            Dropout probability applied after activation layers (default: 0.0).
        batch_norm : bool, optional
            If True, insert batch normalization after linear layers (default: False).
        """

        super(DeepSurvFFNN, self).__init__()

        activation_fn = _get_activation(activation)
        layers, output_size = _build_hidden_layers(number_inputs, hidden_layers, activation_fn, dropout, batch_norm)
        self.layers = nn.ModuleList(layers)

        # Output layer (log hazard ratio)
        self.output_layer = nn.Linear(output_size, 1)

    def forward(self, x):

        """
        Compute the forward pass of the DeepSurv network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, number_inputs)` containing the
            input features for each sample.

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
            Output tensor containing the log hazard ratio for each sample.
        """

        for layer in self.layers:
            x = layer(x)

        return self.output_layer(x)


class DeepMultiTaskFFNN(nn.Module):

    """
    Neural network architecture for DeepMultiTask.
    """

    def __init__(self, number_inputs, number_events=None, hidden_layers=None, activation="relu", dropout=0.0, batch_norm=False):

        """
        Initialize the DeepMultiTask feed-forward neural network.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        number_events : int, optional
            Number of competing events / tasks for which the model predicts hazards.
        hidden_layers : list or None, optional
            Sizes of hidden layers (default: None).
        activation : str, optional
            Activation function name to use for hidden layers (default: "relu").
        dropout : float, optional
            Dropout probability applied after activation layers (default: 0.0).
        batch_norm : bool, optional
            If True, insert batch normalization after linear layers (default: False).
        """

        super(DeepMultiTaskFFNN, self).__init__()

        activation_fn = _get_activation(activation)
        layers, output_size = _build_hidden_layers(number_inputs, hidden_layers, activation_fn, dropout, batch_norm)
        self.layers = nn.ModuleList(layers)

        # Output layer (log hazard ratio per task)
        self.cox_output = nn.Linear(output_size, number_events)

    def forward(self, x):

        """
        Compute the forward pass of the DeepMultiTask network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, number_inputs)` containing the
            input features for each sample.

        Returns
        -------
        tuple
            A tuple `(shared_representation, cox_logits)` where `shared_representation`
            is the output of the shared layers (torch.Tensor of shape
            `(batch_size, feature_dim)`) and `cox_logits` is a tensor of shape
            `(batch_size, number_events)` containing the per-task logit outputs
            (log hazard ratios) for each sample.
        """

        for layer in self.layers:
            x = layer(x)

        return x, self.cox_output(x)

class DeepHitFFNN(nn.Module):

    """
    Neural network architecture for DeepHit.
    """

    def __init__(self, number_inputs, number_events, number_categories, hidden_layers_shared=None, hidden_layers_specific=None, activation="relu", dropout=0.0, batch_norm=False):

        """
        Initialize the DeepHit feed-forward neural network.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        number_events : int
            Number of competing events.
        number_categories : int
            Number of discrete time categories for the survival distribution.
        hidden_layers_shared : list or None, optional
            Sizes of hidden layers for the shared sub-network (default: None).
        hidden_layers_specific : list or None, optional
            Sizes of hidden layers for each event-specific sub-network (default: None).
        activation : str, optional
            Activation function name to use for hidden and specific layers (default: "relu").
        dropout : float, optional
            Dropout probability applied after activation layers (default: 0.0).
        batch_norm : bool, optional
            If True, insert batch normalization after linear layers (default: False).
        """
        
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

        # Build one specific sub-network per competitive events
        self.specific_nets = nn.ModuleList()
        for _ in range(self.number_events):
            specific_layers, specific_output_size = _build_hidden_layers(specific_input_size, hidden_layers_specific, activation_fn, dropout, batch_norm)
            self.specific_nets.append(nn.Sequential(*specific_layers))

        # Output specific sub-networks dimension
        self.specific_output_dimension = specific_output_size

        # Final output layer
        self.output_layer = nn.Linear(self.specific_output_dimension * self.number_events, self.number_categories * self.number_events)

    def forward(self, x):

        """
        Compute the forward pass of the DeepHit network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, number_inputs)` containing the
            input features for each sample.

        Returns
        -------
        torch.Tensor of shape (batch_size, number_events, number_categories)
            Tensor containing the probability distribution over discrete time
            categories for each competing event. Values are probabilities
            (softmax-normalized across categories for each event).
        """

        shared_output = self.shared_net(x)
        specific_input = torch.cat([x, shared_output], dim=1) if self.shared_output_dimension > 0 else x

        specific_outputs = [net(specific_input) for net in self.specific_nets]
        specific_outputs = torch.stack(specific_outputs, dim=1)
        specific_outputs = specific_outputs.reshape(specific_outputs.shape[0], -1)

        output = self.output_layer(specific_outputs)

        # Phantom logit
        zero_logit = torch.zeros(output.shape[0], 1, device=output.device, dtype=output.dtype)
        output = torch.cat([output, zero_logit], dim=1)

        output = torch.softmax(output, dim=1)

        output = output[:, :-1]
        output = output.reshape(-1, self.number_events, self.number_categories)

        return output