import torch
import torch.nn as nn
import timm
from library.config import Config


class EEGNet(nn.Module):
    """
    Channel-Depth Stacked 2D CNN with a Spatial Adapter.

    This model treats the 19 EEG channels as depth channels in a 2D image.
    A learnable 1x1 convolution (Spatial Adapter) mixes these 19 channels
    down to 3 channels, preserving spatial independence while making the
    input compatible with standard ImageNet-pretrained backbones.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load ImageNet weights for the backbone.
        """
        super(EEGNet, self).__init__()

        # ==========================================
        # 1. Spatial Adapter (Spatial Mixing)
        # ==========================================
        # Input: (Batch, 19, Height, Width)
        # Output: (Batch, 3, Height, Width)
        # The 1x1 kernel mixes information across the 19 electrodes (depth)
        # without mixing information across time or frequency (spatial).
        self.spatial_adapter = nn.Conv2d(
            in_channels=Config.N_CHANNELS,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # ==========================================
        # 2. Backbone (Feature Extractor)
        # ==========================================
        # EfficientNet-B2 pretrained on ImageNet.
        # num_classes=0 removes the default classifier.
        # global_pool='avg' ensures we get a pooled feature vector (B, num_features).
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # ==========================================
        # 3. Classification Head
        # ==========================================
        self.num_features = self.backbone.num_features

        self.dropout = nn.Dropout(p=Config.DROPOUT)
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 19, Height, Width).
                              Represents 19-channel spectrograms.

        Returns:
            torch.Tensor: Predicted probabilities of shape (Batch, 6).
        """
        # 1. Adapt dimensions: (B, 19, H, W) -> (B, 3, H, W)
        x = self.spatial_adapter(x)

        # 2. Extract features: (B, 3, H, W) -> (B, num_features)
        x = self.backbone(x)

        # 3. Classify: (B, num_features) -> (B, 6)
        x = self.dropout(x)
        logits = self.fc(x)

        # Output probabilities (Softmax) as required for KL Divergence
        return torch.softmax(logits, dim=1)
