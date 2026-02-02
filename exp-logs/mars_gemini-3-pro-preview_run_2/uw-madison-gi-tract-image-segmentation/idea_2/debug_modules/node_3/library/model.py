import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) block for the SegFormer decoder.
    In the context of SegFormer, this is effectively a linear projection
    of the channel dimension. We use a 1x1 Convolution for efficient
    processing of spatial tensors (B, C, H, W).
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.

    Architecture:
        - Backbone: MiT-B0 (Mix Transformer) from `timm`
        - Decoder: All-MLP Decoder

    Args:
        backbone_name (str): Name of the timm model to use (default: 'mit_b0').
        num_classes (int): Number of output segmentation classes.
        embedding_dim (int): Common channel dimension for the decoder MLP layers.
        pretrained (bool): Whether to load pretrained ImageNet weights for the backbone.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        embedding_dim=256,
        pretrained=True,
    ):
        super().__init__()

        # 1. Backbone (MiT-B0)
        # features_only=True ensures the model returns a list of feature maps
        # from different stages rather than a single classification output.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Extract channel counts from the backbone feature info
        # For mit_b0, this is typically [32, 64, 160, 256]
        self.encoder_channels = self.backbone.feature_info.channels()
        if len(self.encoder_channels) == 5:
            self.encoder_channels = self.encoder_channels[1:]

        # 2. Decoder (All-MLP)
        # We define a projection layer for each of the 4 encoder stages
        self.mlp_c1 = MLP(self.encoder_channels[0], embedding_dim)
        self.mlp_c2 = MLP(self.encoder_channels[1], embedding_dim)
        self.mlp_c3 = MLP(self.encoder_channels[2], embedding_dim)
        self.mlp_c4 = MLP(self.encoder_channels[3], embedding_dim)

        # Fusion layer: takes concatenated features (4 * embedding_dim) and fuses them
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # 3. Classification Head
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes, H, W).
        """
        input_shape = x.shape[-2:]

        # --- Encoder ---
        features = self.backbone(x)
        if len(features) == 5:
            c1, c2, c3, c4 = features[1:]
        else:
            c1, c2, c3, c4 = features

        # --- Decoder ---

        # 1. MLP Layer: Project all multi-scale features to a common embedding_dim
        c1 = self.mlp_c1(c1)  # Shape: (B, 256, H/4, W/4)
        c2 = self.mlp_c2(c2)  # Shape: (B, 256, H/8, W/8)
        c3 = self.mlp_c3(c3)  # Shape: (B, 256, H/16, W/16)
        c4 = self.mlp_c4(c4)  # Shape: (B, 256, H/32, W/32)

        # 2. Upsample Layer: Upsample all features to the resolution of the largest feature map (c1)
        # We use bilinear interpolation. align_corners=False is standard for segmentation.
        target_size = c1.shape[2:]

        c2 = F.interpolate(c2, size=target_size, mode="bilinear", align_corners=False)
        c3 = F.interpolate(c3, size=target_size, mode="bilinear", align_corners=False)
        c4 = F.interpolate(c4, size=target_size, mode="bilinear", align_corners=False)

        # 3. Concatenation Layer
        # Concatenate along the channel dimension
        fused_features = torch.cat(
            [c4, c3, c2, c1], dim=1
        )  # Shape: (B, 1024, H/4, W/4)

        # 4. Fusion MLP
        x = self.linear_fuse(fused_features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 5. Classification Head
        logits = self.classifier(x)  # Shape: (B, num_classes, H/4, W/4)

        # 6. Final Upsample to original input resolution
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )

        return logits
