import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import math
from library.config import Config


class DeformableConvBlock(nn.Module):
    """
    A block containing a Deformable Convolution v2, followed by BN and ReLU.
    Predicts its own offsets and modulation masks.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DeformableConvBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Convolution to predict offsets (2 * k * k) and masks (k * k)
        # We group them into one conv layer for efficiency: 3 * k * k output channels
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        # The learnable weights for the deformable convolution
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offset/mask conv
        # Offsets should be close to 0, masks (sigmoid) should be around 0.5 (bias 0)
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )

        x = self.bn(x)
        x = self.relu(x)
        return x


class SwinCenterNet(nn.Module):
    def __init__(self):
        super(SwinCenterNet, self).__init__()

        # 1. Backbone: Swin Transformer Base
        # features_only=True returns a list of feature maps
        # Swin-Base channels: [128, 256, 512, 1024] for strides [4, 8, 16, 32]
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            img_size=Config.IMG_SIZE,
        )

        # Channel dimensions for Swin-Base
        self.in_channels = [128, 256, 512, 1024]
        self.fpn_dim = 256

        # 2. Neck: FPN (Top-Down Pathway)
        # Lateral convolutions to project to FPN dimension
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, self.fpn_dim, kernel_size=1) for c in self.in_channels]
        )

        # 3. Heads
        # We use a shared Deformable Conv layer for each head before the final projection
        self.head_conv = 256

        # Heatmap Head
        self.hm_head = nn.Sequential(
            DeformableConvBlock(self.fpn_dim, self.head_conv),
            nn.Conv2d(self.head_conv, 1, kernel_size=1, bias=True),
        )

        # WH Head (Width, Height)
        self.wh_head = nn.Sequential(
            DeformableConvBlock(self.fpn_dim, self.head_conv),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, bias=True),
        )

        # Regression Head (Offset X, Offset Y)
        self.reg_head = nn.Sequential(
            DeformableConvBlock(self.fpn_dim, self.head_conv),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, bias=True),
        )

        # Classification Head
        self.cls_head = nn.Sequential(
            DeformableConvBlock(self.fpn_dim, self.head_conv),
            nn.Conv2d(self.head_conv, Config.NUM_CLASSES, kernel_size=1, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize Heatmap Head bias for Focal Loss
        # bias = -log((1 - pi) / pi) where pi = 0.1
        self.hm_head[-1].bias.data.fill_(-2.19)

        # Initialize other heads
        for head in [self.wh_head, self.reg_head, self.cls_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    # Skip the DeformableConvBlock internals as they are already init
                    if m.kernel_size == (1, 1):
                        nn.init.normal_(m.weight, std=0.001)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone feature extraction
        # feats indices: 0->stride 4, 1->stride 8, 2->stride 16, 3->stride 32
        feats = self.backbone(x)

        # Swin Transformer returns features in (B, H, W, C) format.
        # We need to permute them to (B, C, H, W) for Conv2d layers.
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        # FPN Top-Down Pathway
        # Start from the deepest layer
        p3 = self.lateral_convs[3](feats[3])

        # Upsample and add to stride 16
        p2 = self.lateral_convs[2](feats[2]) + F.interpolate(
            p3, scale_factor=2, mode="nearest"
        )

        # Upsample and add to stride 8
        p1 = self.lateral_convs[1](feats[1]) + F.interpolate(
            p2, scale_factor=2, mode="nearest"
        )

        # Upsample and add to stride 4
        p0 = self.lateral_convs[0](feats[0]) + F.interpolate(
            p1, scale_factor=2, mode="nearest"
        )

        # p0 is now the fused feature map at stride 4

        # Heads
        hm = self.hm_head(p0)
        wh = self.wh_head(p0)
        reg = self.reg_head(p0)
        cls_logits = self.cls_head(p0)

        # Apply sigmoid to heatmap
        hm = torch.sigmoid(hm)

        # Note: cls_logits are raw logits, CrossEntropyLoss will handle softmax/log_softmax
        # or we can apply sigmoid if using BCE. The loss function in loss.py uses CrossEntropyLoss
        # which expects logits.

        return {"hm": hm, "wh": wh, "reg": reg, "cls": cls_logits}
