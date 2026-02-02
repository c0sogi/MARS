import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers in a module with GroupNorm layers.

    Args:
        module (nn.Module): The module to modify.
        num_groups (int): The number of groups for GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_channels is divisible by num_groups
            # If channels < groups, we adjust groups to match channels (edge case safety)
            groups = num_groups if num_channels % num_groups == 0 else num_channels

            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels)

            # Replace the layer
            setattr(module, name, gn)
        else:
            # Recurse into children
            replace_bn_with_gn(child, num_groups)


def get_group_norm_resnet34(pretrained=True):
    """
    Instantiates a ResNet34 model and replaces BN with GN.

    Args:
        pretrained (bool): Whether to load ImageNet weights.

    Returns:
        nn.Module: Modified ResNet34.
    """
    # Handle torchvision version differences for weights
    try:
        from torchvision.models import ResNet34_Weights

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    except ImportError:
        model = models.resnet34(pretrained=pretrained)

    # Replace Batch Normalization with Group Normalization
    replace_bn_with_gn(model, num_groups=Config.GROUPS)

    return model


class CervicalMILModel(nn.Module):
    """
    Group-Normalized Deep Multi-Task MIL Network.

    Architecture:
    1. Backbone: ResNet34 with GroupNorm (extracts features per slice).
    2. Context: 1D Conv + GN + ReLU (processes sequence of slices).
    3. Aggregation: Global Max Pooling (aggregates evidence).
    4. Heads: 8 Independent Classifiers (C1-C7, Patient Overall).
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(CervicalMILModel, self).__init__()

        # 1. Backbone
        self.backbone = get_group_norm_resnet34(pretrained=pretrained)

        # Extract layers to form feature extractor (remove avgpool and fc)
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        # ResNet34 layer4 outputs 512 channels
        self.feature_dim = 512

        # Spatial pooling to reduce (C, H, W) -> (C, 1, 1) per slice
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Context Module
        # Input: (B, feature_dim, num_slices)
        # Structure: Conv1d -> GroupNorm -> ReLU
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.context_gn = nn.GroupNorm(
            num_groups=Config.GROUPS, num_channels=self.feature_dim
        )
        self.context_act = nn.ReLU(inplace=True)

        # 3. Multi-Task Heads
        # 8 distinct heads for C1..C7 and Patient_Overall
        self.heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, 1) for _ in range(num_classes)]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, H, W).
                              e.g., (B, 64, 3, 256, 256)

        Returns:
            torch.Tensor: Logits of shape (Batch, 8).
        """
        b, s, c, h, w = x.shape

        # 1. Feature Extraction (Backbone)
        # Merge batch and sequence dims: (B*S, C, H, W)
        x = x.view(b * s, c, h, w)

        # Pass through ResNet backbone
        features = self.feature_extractor(x)  # -> (B*S, 512, H', W')

        # Spatial Pooling
        features = self.spatial_pool(features)  # -> (B*S, 512, 1, 1)
        features = features.flatten(1)  # -> (B*S, 512)

        # 2. Context Processing
        # Reshape to (B, Feature_Dim, Slices) for Conv1d
        features = features.view(b, s, self.feature_dim)
        features = features.permute(0, 2, 1)  # -> (B, 512, S)

        # Apply Context Module
        features = self.context_conv(features)
        features = self.context_gn(features)
        features = self.context_act(features)

        # 3. Aggregation (Global Max Pooling)
        # Max pool over the sequence dimension (dim=2)
        # (B, 512, S) -> (B, 512)
        features = torch.max(features, dim=2)[0]

        # 4. Multi-Task Heads
        # Apply each head independently
        outputs = []
        for head in self.heads:
            # head(features) -> (B, 1)
            outputs.append(head(features))

        # Concatenate outputs: (B, 8)
        logits = torch.cat(outputs, dim=1)

        return logits
