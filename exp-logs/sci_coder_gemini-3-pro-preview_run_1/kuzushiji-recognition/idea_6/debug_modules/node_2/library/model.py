import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import math
from library.config import NUM_CLASSES, BACKBONE


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 Layer.
    Learns offsets and masks to adaptively sample from the input feature map.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Standard convolution weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Offset and Mask generator
        # Output channels:
        #   2 * kernel_size^2 (offsets x, y)
        #   + 1 * kernel_size^2 (modulation mask)
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to standard Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offsets and masks to 0
        # This ensures the layer starts as a standard convolution
        # sigmoid(0) = 0.5 for mask
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )


class Head(nn.Module):
    """
    Standard detection head with a Deformable Convolution followed by a 1x1 projection.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=256, init_bias=None):
        super(Head, self).__init__()
        self.dcn = DeformableConv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Custom bias initialization (useful for focal loss stability)
        if init_bias is not None:
            self.out_conv.bias.data.fill_(init_bias)
        else:
            nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x):
        x = self.dcn(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.out_conv(x)
        return x


class ConvNextCenterNet(nn.Module):
    """
    CenterNet architecture with ConvNeXt-Tiny backbone, FPN, and Deformable Heads.
    """

    def __init__(self):
        super(ConvNextCenterNet, self).__init__()

        # 1. Backbone: ConvNeXt Tiny
        # features_only=True returns a list of feature maps at different strides
        self.backbone = timm.create_model(BACKBONE, pretrained=True, features_only=True)

        # Get channel counts from backbone (typically [96, 192, 384, 768])
        feature_info = self.backbone.feature_info.info
        in_channels_list = [x["num_chs"] for x in feature_info]

        # 2. Neck: Feature Pyramid Network (FPN)
        # Projects all levels to 256 channels
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list, out_channels=256
        )

        # 3. Heads
        # Heatmap Head: Class Agnostic (1 channel), init bias for focal loss
        self.hm_head = Head(256, 1, init_bias=-2.19)

        # WH Head: Width, Height (2 channels)
        self.wh_head = Head(256, 2, init_bias=None)

        # Regression Head: Offset X, Offset Y (2 channels)
        self.reg_head = Head(256, 2, init_bias=None)

        # Classification Head: Logits for all classes
        # Initialized to 0 as we use CrossEntropy on specific points
        self.cls_head = Head(256, NUM_CLASSES, init_bias=0.0)

    def forward(self, x):
        # Backbone Forward
        features = self.backbone(x)

        # FPN Forward
        # FPN expects a dictionary of feature maps
        features_dict = {str(i): f for i, f in enumerate(features)}
        fpn_out = self.fpn(features_dict)

        # Upsample and Aggregate
        # Target resolution is the finest stride (stride 4, key '0')
        p0 = fpn_out["0"]  # Stride 4
        p1 = fpn_out["1"]  # Stride 8
        p2 = fpn_out["2"]  # Stride 16
        p3 = fpn_out["3"]  # Stride 32

        target_h, target_w = p0.shape[2], p0.shape[3]

        # Bilinear upsampling to match P0 resolution
        p1_up = F.interpolate(
            p1, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        p2_up = F.interpolate(
            p2, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        p3_up = F.interpolate(
            p3, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        # Fuse features by summation
        x_fused = p0 + p1_up + p2_up + p3_up

        # Heads Forward
        hm = torch.sigmoid(self.hm_head(x_fused))
        wh = self.wh_head(x_fused)
        reg = self.reg_head(x_fused)
        cls_logits = self.cls_head(x_fused)

        return {"hm": hm, "wh": wh, "reg": reg, "cls_logits": cls_logits}
