import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class HybridResNet34(nn.Module):
    """
    A Hybrid Neural Network that fuses a Spectrogram Encoder (ResNet34)
    with a Statistical Feature Encoder (MLP).
    """

    def __init__(self, num_stats_features=140):
        """
        Args:
            num_stats_features (int): Number of input statistical features.
                                      Default is 140 (14 features * 10 sensors).
        """
        super(HybridResNet34, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Spectrogram Encoder (ResNet34 Backbone)
        # ---------------------------------------------------------------------
        # Load pretrained weights
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 10 channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1

        # Create new layer with 10 input channels
        new_conv1 = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Initialize weights: Average original RGB weights and replicate
        # This preserves the spatial filters learned from ImageNet while adapting to 10 channels
        with torch.no_grad():
            # original_conv1.weight shape: [64, 3, 7, 7]
            w_orig = original_conv1.weight
            # Average over channel dimension -> [64, 1, 7, 7]
            w_mean = torch.mean(w_orig, dim=1, keepdim=True)
            # Replicate for new input channels -> [64, 10, 7, 7]
            w_new = w_mean.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv1.weight.copy_(w_new)

        self.backbone.conv1 = new_conv1

        # Remove the original Fully Connected (FC) layer
        # We want the output of the Global Average Pooling (GAP) layer
        # ResNet18 GAP output dimension is 512
        self.backbone_out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ---------------------------------------------------------------------
        # 2. Statistical Feature Encoder (MLP)
        # ---------------------------------------------------------------------
        # Processes the vector of engineered statistics
        # Architecture: Input -> 128 -> 64
        self.mlp_out_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(num_stats_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, self.mlp_out_dim),
            nn.BatchNorm1d(self.mlp_out_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # 3. Fusion Head
        # ---------------------------------------------------------------------
        # Concatenate backbone embedding and MLP embedding
        fusion_dim = self.backbone_out_dim + self.mlp_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, Config.FC_DIM),
            nn.BatchNorm1d(Config.FC_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FC_DIM, 1),  # Single regression output
        )

    def forward(self, spectrogram, features):
        """
        Forward pass of the hybrid model.

        Args:
            spectrogram (torch.Tensor): Stacked multi-resolution spectrograms.
                                        Shape: (Batch, 20, 128, Time)
            features (torch.Tensor): Statistical feature vector.
                                     Shape: (Batch, num_stats_features)

        Returns:
            torch.Tensor: Predicted scaled time_to_eruption. Shape: (Batch, 1)
        """
        # --- Branch 1: Spectrogram Encoder ---
        # Pass through ResNet34 backbone
        # Output shape: (Batch, 512) (after GAP and Flatten via Identity fc)
        spec_embedding = self.backbone(spectrogram)

        # --- Branch 2: Statistical Encoder ---
        # Pass through MLP
        # Output shape: (Batch, 64)
        stats_embedding = self.mlp(features)

        # --- Fusion ---
        # Concatenate embeddings along the feature dimension
        # Shape: (Batch, 512 + 64) = (Batch, 576)
        combined = torch.cat((spec_embedding, stats_embedding), dim=1)

        # --- Prediction ---
        # Pass through regression head
        # Shape: (Batch, 1)
        output = self.head(combined)

        return output
