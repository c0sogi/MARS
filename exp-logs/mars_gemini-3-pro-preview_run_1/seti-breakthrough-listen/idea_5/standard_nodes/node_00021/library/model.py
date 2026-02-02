import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library import config


def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_groups divides num_channels
            groups = num_groups
            if num_channels % groups != 0:
                groups = 1  # Fallback if channels are too few

            gn = nn.GroupNorm(groups, num_channels)

            # Transfer affine parameters (weight/bias) if they exist
            # Note: Running stats (mean/var) are discarded as GN doesn't use them
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)

            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


class TimeDistributedResNet50GN(nn.Module):
    def __init__(self):
        super(TimeDistributedResNet50GN, self).__init__()

        # 1. Load Pretrained Backbone
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            backbone = models.resnet50(weights=weights)
        except:
            # Fallback for older torchvision versions
            backbone = models.resnet50(pretrained=True)

        # 2. Modify Input Layer for 1 Channel (Spectrogram)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize the single channel by averaging the original RGB weights
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)

        backbone.conv1 = new_conv

        # 3. Replace BatchNorm with GroupNorm for stability at small batch sizes
        if config.USE_GROUP_NORM:
            replace_bn_with_gn(backbone, num_groups=config.GN_GROUPS)

        # 4. Deconstruct Backbone for Multi-Scale Access
        # We retain the stem and layers 1-4
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # Stride 4
        self.layer2 = backbone.layer2  # Stride 8, 512 ch
        self.layer3 = backbone.layer3  # Stride 16, 1024 ch
        self.layer4 = backbone.layer4  # Stride 32, 2048 ch

        # 5. FPN / Late Fusion Layers
        # We fuse features from all 6 time steps by concatenating channels
        fpn_dim = config.FPN_CHANNELS

        # Lateral layers to project fused features (Time * Channels) to FPN dimension
        self.lat_layer4 = nn.Conv2d(2048 * 6, fpn_dim, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(1024 * 6, fpn_dim, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(512 * 6, fpn_dim, kernel_size=1)

        # 6. Classification Head
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc = nn.Linear(fpn_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 6, 1, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        b, t, c, h, w = x.shape

        # Time-Distributed: Fold time into batch dimension
        # (B, 6, 1, H, W) -> (B*6, 1, H, W)
        x = x.view(b * t, c, h, w)

        # Backbone Forward Pass
        x = self.stem(x)  # Stride 4
        c1 = self.layer1(x)  # Stride 4
        c2 = self.layer2(c1)  # Stride 8
        c3 = self.layer3(c2)  # Stride 16
        c4 = self.layer4(c3)  # Stride 32

        # Helper function to reshape and fuse time steps
        def fuse_time(feat):
            # feat: (B*T, C_feat, H_feat, W_feat)
            _, ch, fh, fw = feat.shape

            # Unfold time dimension: (B, T, C_feat, H_feat, W_feat)
            feat = feat.view(b, t, ch, fh, fw)

            # Permute to stack channels: (B, T, C, H, W) -> (B, T*C, H, W)
            # We treat features from different time steps as different channels
            feat = feat.permute(0, 1, 2, 3, 4).contiguous()
            feat = feat.view(b, t * ch, fh, fw)
            return feat

        # Fuse features from all time steps
        f4 = fuse_time(c4)  # (B, 12288, H/32, W/32)
        f3 = fuse_time(c3)  # (B, 6144, H/16, W/16)
        f2 = fuse_time(c2)  # (B, 3072, H/8, W/8)

        # Feature Pyramid Network (Top-Down Pathway)
        # 1. Project deepest layer
        p4 = self.lat_layer4(f4)

        # 2. Project middle layer and add upsampled deep layer
        p3_in = self.lat_layer3(f3)
        p3 = p3_in + F.interpolate(p4, size=p3_in.shape[-2:], mode="nearest")

        # 3. Project shallow layer and add upsampled middle layer
        p2_in = self.lat_layer2(f2)
        p2 = p2_in + F.interpolate(p3, size=p2_in.shape[-2:], mode="nearest")

        # Global Average Pooling on the highest resolution FPN feature map (P2)
        out = F.adaptive_avg_pool2d(p2, (1, 1))
        out = torch.flatten(out, 1)

        # Classification Head
        out = self.dropout(out)
        out = self.fc(out)

        return out
