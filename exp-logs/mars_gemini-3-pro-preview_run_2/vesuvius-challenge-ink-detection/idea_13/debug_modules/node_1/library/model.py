import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class MLP(nn.Module):
    """
    Simple MLP module for SegFormer decoder.
    Uses 1x1 Conv2d which is equivalent to a Linear layer applied to each pixel.
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    """
    A Conv2d -> BatchNorm -> ReLU block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SiameseSegFormer(nn.Module):
    """
    Siamese Multi-View SegFormer (Siamese-MVS).

    Architecture:
    1. Shared Encoder (MiT-B2): Processes 3 views (High, Center, Low) independently.
    2. Feature Fusion: Element-wise Max Pooling across views at each scale.
    3. Decoder: MLP-based decoder fusing multi-scale features to predict ink.
    """

    def __init__(self):
        super().__init__()

        # 1. Shared Encoder (MiT-B2)
        # features_only=True returns a list of feature maps from different stages
        self.encoder = timm.create_model(
            Config.ENCODER_NAME, pretrained=True, features_only=True
        )

        # MiT-B2 feature channels: [64, 128, 320, 512]
        # We retrieve these dynamically to be safe
        encoder_channels = self.encoder.feature_info.channels()

        # 2. Decoder Hyperparameters
        embedding_dim = 768  # Standard for SegFormer-B2

        # 3. Decoder Layers
        # Projections for each scale
        self.linear_c1 = MLP(encoder_channels[0], embedding_dim)
        self.linear_c2 = MLP(encoder_channels[1], embedding_dim)
        self.linear_c3 = MLP(encoder_channels[2], embedding_dim)
        self.linear_c4 = MLP(encoder_channels[3], embedding_dim)

        # Fusion layer
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1
        )

        self.dropout = nn.Dropout(0.1)

        # Classifier
        self.classifier = nn.Conv2d(embedding_dim, Config.NUM_CLASSES, kernel_size=1)

        # 4. Normalization (ImageNet stats)
        # Input is [0, 1], encoder expects standardized input
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward_encoder(self, x):
        """
        Normalizes input and passes through the backbone.
        """
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, x_high, x_center, x_low):
        """
        Args:
            x_high: (B, 3, H, W) - Upper Z-slice range
            x_center: (B, 3, H, W) - Center Z-slice range
            x_low: (B, 3, H, W) - Lower Z-slice range
        Returns:
            logits: (B, 1, H, W)
        """
        # --- 1. Siamese Encoding ---
        # Extract features for each view using shared weights
        # features is a list of tensors: [c1, c2, c3, c4]
        feats_h = self.forward_encoder(x_high)
        feats_c = self.forward_encoder(x_center)
        feats_l = self.forward_encoder(x_low)

        # --- 2. Feature Fusion (Element-wise Max Pooling) ---
        # For each scale, take the max activation across the 3 views
        feats_fused = []
        for i in range(len(feats_h)):
            # Stack along new dim 0 -> (3, B, C, H, W)
            stacked = torch.stack([feats_h[i], feats_c[i], feats_l[i]], dim=0)
            # Max over dim 0 -> (B, C, H, W)
            f_max, _ = torch.max(stacked, dim=0)
            feats_fused.append(f_max)

        c1, c2, c3, c4 = feats_fused

        # --- 3. MLP Decoder ---

        # Get shape of the largest feature map (c1) which is 1/4th of input
        # We upsample everything to this size
        h, w = c1.shape[2], c1.shape[3]

        # Process C4 (1/32) -> 1/4
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode="bilinear", align_corners=False)

        # Process C3 (1/16) -> 1/4
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode="bilinear", align_corners=False)

        # Process C2 (1/8) -> 1/4
        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode="bilinear", align_corners=False)

        # Process C1 (1/4)
        _c1 = self.linear_c1(c1)

        # Concatenate
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        # Fuse and Classify
        x = self.linear_fuse(_c)
        x = self.dropout(x)
        x = self.classifier(x)

        # --- 4. Final Upsampling ---
        # Upsample from 1/4 to Full Resolution (512x512)
        # Use x_high to get target dimensions
        x = F.interpolate(
            x, size=x_high.shape[2:], mode="bilinear", align_corners=False
        )

        return x
