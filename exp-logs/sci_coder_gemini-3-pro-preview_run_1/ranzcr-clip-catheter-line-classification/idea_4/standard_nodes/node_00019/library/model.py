import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (mean(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN in pow if input is negative (e.g., from SiLU/Swish activations)
        # We avoid explicit casting to FP32 to maintain Mixed Precision performance where possible,
        # relying on PyTorch's stability.
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class SegmentationDecoder(nn.Module):
    """
    Lightweight FPN-style decoder for the auxiliary segmentation task.
    Aggregates features from multiple backbone stages to produce a binary mask.
    """

    def __init__(self, feature_channels, out_channels=1, hidden_dim=128):
        super().__init__()
        # We typically use features from strides 4, 8, 16, 32
        # These correspond to indices 1, 2, 3, 4 in EfficientNet features (index 0 is stride 2)
        self.used_indices = [1, 2, 3, 4]

        # Safety check for backbones with fewer levels
        if len(feature_channels) < 5:
            self.used_indices = list(range(len(feature_channels)))[-4:]

        self.conv_projects = nn.ModuleList()
        for idx in self.used_indices:
            in_ch = feature_channels[idx]
            self.conv_projects.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, features):
        """
        Args:
            features (list): List of feature maps from the backbone.
        Returns:
            torch.Tensor: Segmentation logits.
        """
        # Top-down pathway: Start from the deepest feature
        x = self.conv_projects[-1](features[self.used_indices[-1]])

        # Iterate backwards through the selected levels
        for i in range(len(self.used_indices) - 2, -1, -1):
            feat_idx = self.used_indices[i]
            proj = self.conv_projects[i](features[feat_idx])

            # Upsample x to match the spatial size of the current projection
            if x.shape[-2:] != proj.shape[-2:]:
                x = F.interpolate(
                    x, size=proj.shape[-2:], mode="bilinear", align_corners=False
                )

            x = x + proj

        # Final convolution
        x = self.final_conv(x)

        # Upsample to approximate original image size (stride 4 -> stride 1)
        # Note: The main model forward pass ensures exact size matching.
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x


class MultiTaskEfficientNet(nn.Module):
    """
    EfficientNetV2-S backbone with Multi-Task Learning heads:
    1. Classification Head: GeM Pooling -> Linear Layer -> 11 Class Logits
    2. Auxiliary Segmentation Head: FPN Decoder -> Binary Mask Logits
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super().__init__()

        # 1. Backbone
        # features_only=True returns a list of feature maps from different stages
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        feature_channels = self.backbone.feature_info.channels()

        # 2. Classification Head
        # Uses the last feature map (typically stride 32)
        last_channel = feature_channels[-1]
        self.gem = GeM()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(last_channel, num_classes)

        # 3. Segmentation Head (Auxiliary)
        self.seg_decoder = SegmentationDecoder(feature_channels)

    def forward(self, x):
        # x: (B, 3, H, W)

        # Extract features from backbone
        features = self.backbone(x)

        # --- Classification Branch ---
        global_feat = features[-1]
        global_feat = self.gem(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.drop(global_feat)
        cls_logits = self.fc(global_feat)

        # --- Segmentation Branch ---
        seg_logits = self.seg_decoder(features)

        # Ensure seg_logits matches input size exactly (handling potential rounding in downsampling)
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        return cls_logits, seg_logits
