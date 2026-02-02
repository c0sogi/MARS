import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VolcanoHybridModel(nn.Module):
    """
    Hybrid architecture combining a CNN backbone (ResNet18) for spectrograms
    and an MLP for statistical features.

    Key Features:
    1. Modified first conv layer for 10-channel input (Cite Lesson 9).
    2. Global Average Pooling (GAP) (Cite Lesson 15).
    3. Dual-branch fusion.
    """

    def __init__(self):
        super(VolcanoHybridModel, self).__init__()

        # ---------------------------------------------------------
        # 1. Spectrogram Branch (CNN Backbone)
        # ---------------------------------------------------------
        # Load pretrained backbone (ResNet18)
        # global_pool='avg' ensures we get the pooled feature vector [Batch, Channels]
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Modify the first convolution layer to accept 10 channels
        self._modify_first_conv_layer()

        # Determine backbone output channels
        dummy_input = torch.randn(1, Config.IN_CHANNELS, 128, 128)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            backbone_out_channels = features.shape[1]

        # ---------------------------------------------------------
        # 2. Statistical Branch (MLP)
        # ---------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(Config.NUM_STAT_FEATURES, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # ---------------------------------------------------------
        # 3. Fusion Head
        # ---------------------------------------------------------
        # Concatenation of Backbone Output (C) + MLP Output (Hidden)
        fusion_dim = backbone_out_channels + Config.HIDDEN_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
        )

    def _modify_first_conv_layer(self):
        """
        Adapts the first convolutional layer of the backbone to accept
        Config.IN_CHANNELS (10) instead of the default 3 (RGB).

        Weights are initialized by averaging the RGB weights and replicating them.
        """
        # Identify the first layer based on architecture
        if hasattr(self.backbone, "conv1"):
            # ResNet style
            old_conv = self.backbone.conv1
            layer_name = "conv1"
        elif hasattr(self.backbone, "conv_stem"):
            # EfficientNet style
            old_conv = self.backbone.conv_stem
            layer_name = "conv_stem"
        else:
            raise AttributeError(
                "Backbone first layer not found (checked conv1, conv_stem)"
            )

        # Create new conv layer with same parameters but different in_channels
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights
        # Shape: [Out, In, K, K]
        with torch.no_grad():
            # Average the weights across the original 3 channels -> [Out, 1, K, K]
            weight_mean = old_conv.weight.mean(dim=1, keepdim=True)
            # Replicate 10 times -> [Out, 10, K, K]
            new_weight = weight_mean.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv.weight.copy_(new_weight)

            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        setattr(self.backbone, layer_name, new_conv)

    def forward(self, spectrogram, features):
        """
        Forward pass of the hybrid model.

        Args:
            spectrogram (torch.Tensor): [Batch, 10, Freq, Time]
            features (torch.Tensor): [Batch, Num_Stats]

        Returns:
            torch.Tensor: Predicted time to eruption (scaled) [Batch, 1]
        """
        # ---------------------------------------------------------
        # Branch 1: Spectrogram
        # ---------------------------------------------------------
        # Extract pooled features: [Batch, Backbone_Out]
        x_spec_emb = self.backbone(spectrogram)

        # ---------------------------------------------------------
        # Branch 2: Statistical Features
        # ---------------------------------------------------------
        # Process stats: [Batch, Hidden]
        x_stats_emb = self.mlp(features)

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------
        # Concatenate: [Batch, Backbone_Out + Hidden]
        x_fused = torch.cat((x_spec_emb, x_stats_emb), dim=1)

        # Final Prediction
        output = self.head(x_fused)

        return output
