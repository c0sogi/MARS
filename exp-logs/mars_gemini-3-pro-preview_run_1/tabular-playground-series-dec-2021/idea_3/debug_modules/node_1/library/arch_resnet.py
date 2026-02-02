import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A Residual Block adapted for Tabular Data.
    Structure: Input -> Linear -> BN -> Add(Shortcut) -> ReLU -> Dropout
    """

    def __init__(self, in_dim, out_dim, dropout_rate, use_batch_norm):
        super(ResidualBlock, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Shortcut connection:
        # If input and output dimensions differ, use a projection layer.
        # Otherwise, use Identity to pass the input directly.
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.linear(x)
        if self.use_batch_norm:
            out = self.bn(out)

        # Residual connection
        # Add the shortcut (projected or identity) to the main path
        out = out + self.shortcut(x)

        # Activation and Dropout are applied after the addition
        out = self.activation(out)
        out = self.dropout(out)

        return out


class TabularResNet(nn.Module):
    """
    ResNet architecture for Tabular Data.
    Constructs a sequence of ResidualBlocks based on the provided hidden dimensions,
    followed by a final classification head.
    """

    def __init__(
        self, input_dim, num_classes, hidden_dims, dropout_rate=0.0, use_batch_norm=True
    ):
        """
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_dims (list of int): List specifying the size of each hidden layer.
            dropout_rate (float): Probability of dropout.
            use_batch_norm (bool): Whether to use Batch Normalization.
        """
        super(TabularResNet, self).__init__()

        self.layers = nn.ModuleList()
        current_dim = input_dim

        # Build the sequence of residual blocks
        for h_dim in hidden_dims:
            self.layers.append(
                ResidualBlock(
                    in_dim=current_dim,
                    out_dim=h_dim,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                )
            )
            current_dim = h_dim

        # Final output layer producing logits
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        # Pass through all residual blocks
        for layer in self.layers:
            x = layer(x)

        # Pass through classifier
        logits = self.classifier(x)
        return logits
