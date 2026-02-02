import torch
import torch.nn as nn
from library.config import Config


class CrossLayer(nn.Module):
    """
    A single layer of the Cross Network (DCNv2 Matrix formulation).

    Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l

    Where:
        x_0: The original input embedding/features.
        x_l: The output of the previous cross layer.
        W_l, b_l: Learnable weights and biases.
        *: Element-wise multiplication.
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0, xl):
        """
        Args:
            x0 (torch.Tensor): Original input features. Shape (Batch, Input_Dim)
            xl (torch.Tensor): Output from previous layer. Shape (Batch, Input_Dim)

        Returns:
            torch.Tensor: Next layer output. Shape (Batch, Input_Dim)
        """
        # Calculate the interaction term: W_l * x_l + b_l
        interaction = self.linear(xl)

        # Apply the crossing formula: x_0 * interaction + x_l
        return x0 * interaction + xl


class CrossNetwork(nn.Module):
    """
    The Cross Network branch consisting of stacked CrossLayers.
    Captures explicit feature interactions of bounded degree.
    """

    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DeepNetwork(nn.Module):
    """
    The Deep Network branch (MLP).
    Captures implicit high-order non-linearities.
    Structure: [Linear -> BatchNorm -> ReLU -> Dropout] x N
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(DeepNetwork, self).__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.model = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.model(x)


class DCN(nn.Module):
    """
    Ego-Centric Deep Cross Network (EC-DCN).

    Combines a Cross Network and a Deep Network in parallel to capture both
    explicit feature interactions (physics logic) and implicit non-linear patterns.

    Output:
        Raw logits (no sigmoid) for use with BCEWithLogitsLoss.
    """

    def __init__(self):
        super(DCN, self).__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_DIM
        num_cross_layers = Config.NUM_CROSS_LAYERS
        deep_hidden_layers = Config.DEEP_HIDDEN_LAYERS
        dropout_rate = Config.DROPOUT_RATE

        # 1. Cross Branch
        self.cross_net = CrossNetwork(input_dim, num_cross_layers)

        # 2. Deep Branch
        self.deep_net = DeepNetwork(input_dim, deep_hidden_layers, dropout_rate)

        # 3. Combination Layer
        # The Cross Net preserves input dimension.
        # The Deep Net outputs the size of the last hidden layer.
        final_dim = input_dim + self.deep_net.output_dim

        # Final projection to single logit
        self.final_linear = nn.Linear(final_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Flattened wide feature vector. Shape (Batch, Input_Dim)

        Returns:
            torch.Tensor: Logits. Shape (Batch, 1)
        """
        # Parallel execution
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)

        # Concatenate features
        concat = torch.cat([cross_out, deep_out], dim=1)

        # Project to logit
        logits = self.final_linear(concat)

        return logits
