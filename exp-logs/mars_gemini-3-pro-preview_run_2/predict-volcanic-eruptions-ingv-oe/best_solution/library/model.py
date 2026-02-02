import torch
import torch.nn as nn
import timm
from library.config import (
    NUM_SENSORS,
    MLP_HIDDEN_DIM,
    DROPOUT_RATE,
    BACKBONE_NAME,
    PRETRAINED,
)


class HybridResNet(nn.Module):
    """
    Hybrid ResNet18 for Volcanic Eruption Prediction.
    Cite solution_lesson_node_00011: Prioritize Spectral Resolution and Texture over Explicit Sequence Modeling.

    Branch 1: ResNet18 backbone -> Global Average Pooling
    Branch 2: MLP for Statistical Features
    Fusion: Concatenation -> Regression Head
    """

    def __init__(self, num_stats_features=150):
        super(HybridResNet, self).__init__()

        # ------------------------------------------------------------------
        # Branch 1: Spectrogram CNN (ResNet18)
        # ------------------------------------------------------------------
        # Load backbone with global average pooling
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=PRETRAINED, num_classes=0, global_pool="avg"
        )

        # Modify the first layer to accept 10 channels (Sensors)
        self._modify_first_layer(in_channels=NUM_SENSORS)

        # Determine backbone output channels
        with torch.no_grad():
            dummy_spec = torch.randn(1, NUM_SENSORS, 128, 128)
            features = self.backbone(dummy_spec)
            self.cnn_out_dim = features.shape[1]

        # ------------------------------------------------------------------
        # Branch 2: Statistical MLP
        # ------------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(num_stats_features, MLP_HIDDEN_DIM),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM),
            nn.BatchNorm1d(MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # ------------------------------------------------------------------
        # Fusion Head
        # ------------------------------------------------------------------
        fusion_dim = self.cnn_out_dim + MLP_HIDDEN_DIM
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 1),
        )

    def _modify_first_layer(self, in_channels):
        """
        Adapts the first convolutional layer to accept `in_channels`.
        Initializes weights by averaging RGB channels and replicating.
        Cite solution_lesson_node_00009: Preserving Pre-trained Feature Hierarchies.
        """
        # In timm resnet, the first layer is named 'conv1'
        # Check if attribute exists, otherwise try 'conv_stem' (efficientnet)
        if hasattr(self.backbone, "conv1"):
            old_conv = self.backbone.conv1
            layer_name = "conv1"
        elif hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem
            layer_name = "conv_stem"
        else:
            raise AttributeError("Could not find first convolution layer in backbone.")

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Weight Initialization
        with torch.no_grad():
            # old_conv.weight shape: (Out, 3, K, K)
            # Average over the RGB dimension (dim 1)
            weight_avg = old_conv.weight.mean(dim=1, keepdim=True)  # (Out, 1, K, K)
            # Replicate for new input channels
            weight_new = weight_avg.repeat(1, in_channels, 1, 1)  # (Out, 10, K, K)

            new_conv.weight.copy_(weight_new)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        setattr(self.backbone, layer_name, new_conv)

    def forward(self, spec, stats):
        """
        Args:
            spec (torch.Tensor): Spectrogram input (Batch, 10, Freq, Time)
            stats (torch.Tensor): Statistical features (Batch, 150)
        Returns:
            torch.Tensor: Predicted time to eruption (Batch,)
        """
        # --- Spectrogram Branch ---
        # Backbone returns global pooled features (Batch, Features)
        x = self.backbone(spec)

        # --- Statistics Branch ---
        s = self.mlp(stats)  # Shape: (Batch, MLP_Hidden)

        # --- Fusion ---
        combined = torch.cat([x, s], dim=1)
        out = self.regressor(combined)

        return out.squeeze(1)
