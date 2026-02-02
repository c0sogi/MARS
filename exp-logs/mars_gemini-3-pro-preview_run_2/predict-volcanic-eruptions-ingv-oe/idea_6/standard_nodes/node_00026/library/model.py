import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ChannelAdaptiveHybridModel(nn.Module):
    """
    Channel-Adaptive Hybrid ResNet18 Model.

    Architecture:
    1. Branch 1 (Spectrograms): ResNet18 backbone.
       - The first convolution is adapted to accept 10 channels (seismic sensors).
       - Weights are initialized from ImageNet, with the first layer averaged and replicated.
       - Uses Global Average Pooling (GAP) naturally provided by the backbone.
    2. Branch 2 (Statistics): Multi-Layer Perceptron.
       - Processes the vector of statistical features (mean, std, skew, etc.).
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
        # Structure: backbone.conv1 is the first Conv2d layer
        original_conv = self.backbone.conv1

        # Create new Conv2d with IN_CHANNELS (10)
        # We preserve out_channels, kernel_size, stride, padding, and bias settings
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # Initialize weights: Average RGB weights and replicate across 10 channels
        # original_conv.weight shape: [Out, 3, K, K]
        with torch.no_grad():
            w_orig = original_conv.weight
            # Average across the channel dimension (dim 1) -> [Out, 1, K, K]
            w_mean = torch.mean(w_orig, dim=1, keepdim=True)
            # Replicate to match new input channels -> [Out, 10, K, K]
            w_new = w_mean.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv.weight.copy_(w_new)

            # Copy bias if it exists (usually None for ResNet conv1)
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # Replace the layer in the backbone
        self.backbone.conv1 = new_conv

        # Identify the embedding dimension.
        # For ResNet18, the fc input features is 512.
        cnn_out_dim = self.backbone.fc.in_features

        # Replace the classifier with Identity to output the pooled embedding
        # The backbone forward pass includes avgpool and flatten, so this gives us the 1D vector.
        self.backbone.fc = nn.Identity()

        # ---------------------------------------------------------
        # Branch 2: Statistical MLP (Tabular Features)
        # ---------------------------------------------------------
        # Dense -> BN -> ReLU -> Dropout
        self.mlp = nn.Sequential(
            nn.Linear(num_stats_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        mlp_out_dim = 64

        # ---------------------------------------------------------
        # Fusion Head
        # ---------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + mlp_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, Config.NUM_CLASSES),  # Regression output (1)
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
        # EfficientNet expects [Batch, Channels, Height, Width]
        # Our spectrogram is [Batch, 10, Freq, Time], which fits perfectly.
        cnn_features = self.backbone(spectrogram)  # Output: [Batch, 1280]

        # --- Branch 2 ---
        mlp_features = self.mlp(stats)  # Output: [Batch, 64]

        # --- Fusion ---
        combined = torch.cat((cnn_features, mlp_features), dim=1)  # [Batch, 1344]
        output = self.fusion(combined)  # [Batch, 1]

        return output
