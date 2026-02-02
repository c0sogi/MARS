import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    A 1D Inception block that processes input through parallel convolutional branches
    with different kernel sizes to capture multi-scale features.
    """

    def __init__(self, in_channels, out_channels, kernels):
        super(InceptionBlock1D, self).__init__()
        self.branches = nn.ModuleList()

        num_branches = len(kernels)
        # Calculate output channels per branch to sum up to out_channels
        out_ch_per_branch = out_channels // num_branches
        remainder = out_channels % num_branches

        for i, k in enumerate(kernels):
            # Distribute remainder channels to the first few branches
            current_out_ch = out_ch_per_branch + (1 if i < remainder else 0)

            # Branch: Conv1d -> BatchNorm -> ReLU
            # Padding is set to k // 2 to maintain temporal dimension (assuming stride=1)
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    current_out_ch,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(current_out_ch),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

    def forward(self, x):
        # Apply all branches and concatenate along the channel dimension
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class ResidualInceptionLayer(nn.Module):
    """
    Wraps an InceptionBlock1D with a residual connection and max pooling.
    """

    def __init__(
        self, in_channels, out_channels, kernels, pool_kernel=4, pool_stride=4
    ):
        super(ResidualInceptionLayer, self).__init__()

        self.inception = InceptionBlock1D(in_channels, out_channels, kernels)

        # Residual path: Project channels if input/output dimensions differ
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # Downsampling
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        identity = x

        # Main path
        out = self.inception(x)

        # Residual path
        res = self.residual(identity)

        # Combine
        out = out + res
        out = F.relu(out)

        # Downsample
        out = self.pool(out)
        return out


class EEGNet1D(nn.Module):
    """
    Main Multi-Scale 1D CNN architecture for EEG classification.
    """

    def __init__(self, config=Config):
        super(EEGNet1D, self).__init__()

        self.num_classes = config.NUM_CLASSES
        self.kernels = config.KERNELS
        self.hidden_dims = config.HIDDEN_DIMS
        input_channels = config.NUM_CHANNELS

        layers = []
        current_in = input_channels

        # Construct the feature extractor using stacked Residual Inception Layers
        for hidden_dim in self.hidden_dims:
            layer = ResidualInceptionLayer(
                in_channels=current_in,
                out_channels=hidden_dim,
                kernels=self.kernels,
                pool_kernel=4,
                pool_stride=4,
            )
            layers.append(layer)
            current_in = hidden_dim

        self.features = nn.Sequential(*layers)

        # Classification Head
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(self.hidden_dims[-1], self.num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time)
        Returns:
            torch.Tensor: Class probabilities of shape (Batch, Num_Classes)
        """
        # Feature Extraction
        x = self.features(x)

        # Global Average Pooling: Average over the time dimension
        # Shape change: (Batch, Channels, Time) -> (Batch, Channels)
        x = x.mean(dim=2)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        # Output probabilities (Softmax)
        probs = F.softmax(logits, dim=1)
        return probs
