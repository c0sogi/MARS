import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ChannelAdaptiveHybridModel(nn.Module):
    """
    Channel-Adaptive Hybrid ResNet Model.

    Architecture:
    1. Branch 1 (Spectrograms): ResNet18 backbone.
       - The first convolution is adapted to accept 10 channels (seismic sensors).
       - Weights are initialized from ImageNet, with the first layer averaged and replicated.
       - Uses Global Average Pooling (GAP) naturally provided by the backbone.
    2. Branch 2 (Statistics): Multi-Layer Perceptron.
       - Processes the vector of statistical features (mean, std, skew, etc.).
       - High capacity to leverage high-SNR tabular features.
    3. Fusion Head:
       - Concatenates embeddings from both branches.
       - Regresses the final scaled target value.
    """

    def __init__(self, num_stats_features: int):
        """
        Args:
            num_stats_features (int): The number of input features in the statistical vector.
        """
        super().__init__()

        # ---------------------------------------------------------
        # Branch 1: ResNet18 Backbone (Spectrograms)
        # ---------------------------------------------------------
        # Load pre-trained weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)

        # Adapt first Convolutional Layer for 10-channel input
        # Structure: backbone.conv1 is the first layer
        original_conv = self.backbone.conv1

        # Create new Conv2d with IN_CHANNELS (10)
        # We preserve out_channels, kernel_size, stride, padding, and bias settings
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize weights: Average RGB weights and replicate across 10 channels
        # Cite Lesson 00009: Preserving Pre-trained Feature Hierarchies
        # original_conv.weight shape: [64, 3, 7, 7]
        with torch.no_grad():
            w_orig = original_conv.weight
            # Average across the channel dimension (dim 1) -> [64, 1, 7, 7]
            w_mean = torch.mean(w_orig, dim=1, keepdim=True)
            # Replicate to match new input channels -> [64, 10, 7, 7]
            w_new = w_mean.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv.weight.copy_(w_new)

        # Replace the layer in the backbone
        self.backbone.conv1 = new_conv

        # Identify the embedding dimension.
        # For ResNet18, the fc input features is 512.
        cnn_out_dim = self.backbone.fc.in_features

        # Replace the classifier with Identity to output the pooled embedding
        self.backbone.fc = nn.Identity()

        # ---------------------------------------------------------
        # Branch 2: Statistical MLP (Tabular Features)
        # ---------------------------------------------------------
        # Cite Lesson 00026: Do not under-parameterize the tabular branch
        # Increased capacity: [256, 128] hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(num_stats_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        mlp_out_dim = 128

        # ---------------------------------------------------------
        # Fusion Head
        # ---------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + mlp_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, Config.NUM_CLASSES),  # Regression output (1)
        )

    def forward(self, spectrogram, stats):
        """
        Forward pass of the hybrid model.

        Args:
            spectrogram (torch.Tensor): Shape [Batch, 10, Freq, Time]
            stats (torch.Tensor): Shape [Batch, num_stats_features]

        Returns:
            torch.Tensor: Predicted scaled time_to_eruption [Batch, 1]
        """
        # --- Branch 1 ---
        # ResNet expects [Batch, Channels, Height, Width]
        cnn_features = self.backbone(spectrogram)  # Output: [Batch, 512]

        # --- Branch 2 ---
        mlp_features = self.mlp(stats)  # Output: [Batch, 128]

        # --- Fusion ---
        combined = torch.cat((cnn_features, mlp_features), dim=1)  # [Batch, 640]
        output = self.fusion(combined)  # [Batch, 1]

        return output
