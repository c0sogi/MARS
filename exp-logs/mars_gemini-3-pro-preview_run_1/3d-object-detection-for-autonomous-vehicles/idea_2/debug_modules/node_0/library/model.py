import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# ==============================================================================
# Basic Building Blocks
# ==============================================================================


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        dilation=1,
        root_kernel_size=1,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                dilation=dilation,
                root_kernel_size=root_kernel_size,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                dilation=dilation,
                root_kernel_size=root_kernel_size,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, False)
        else:
            self.root = Root(root_dim, out_channels, root_kernel_size, False)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


# ==============================================================================
# DLA Backbone
# ==============================================================================


class DLA(nn.Module):
    def __init__(
        self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        # Initial convolution
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Level 0
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])

        # Level 1
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )

        # Level 2-5 (Trees)
        self.level2 = Tree(
            levels[2], block, channels[1], channels[2], 2, level_root=False
        )
        self.level3 = Tree(
            levels[3], block, channels[2], channels[3], 2, level_root=True
        )
        self.level4 = Tree(
            levels[4], block, channels[3], channels[4], 2, level_root=True
        )
        self.level5 = Tree(
            levels[5], block, channels[4], channels[5], 2, level_root=True
        )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(6):
            if i == 0:
                x = self.level0(x)
            elif i == 1:
                x = self.level1(x)
            elif i == 2:
                x = self.level2(x)
            elif i == 3:
                x = self.level3(x)
            elif i == 4:
                x = self.level4(x)
            elif i == 5:
                x = self.level5(x)
            y.append(x)

        return y


def dla34(pretrained=False, **kwargs):
    model = DLA(
        [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs
    )
    return model


# ==============================================================================
# Upsampling (IDA)
# ==============================================================================


class IDAUp(nn.Module):
    """
    Iterative Deep Aggregation Upsampling.
    Projects features to a common channel dimension and fuses them bottom-up.
    """

    def __init__(self, in_channels_list, out_channels):
        super(IDAUp, self).__init__()
        self.projs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.nodes = nn.ModuleList()

        # We process from deep to shallow.
        # in_channels_list corresponds to [Level 2, Level 3, Level 4, Level 5]
        # Strides: [4, 8, 16, 32]

        for ic in in_channels_list:
            self.projs.append(
                nn.Sequential(
                    nn.Conv2d(ic, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # We need N-1 upsampling steps to fuse N scales
        for _ in range(len(in_channels_list) - 1):
            # Upsample conv (can be Deconv or Conv+Upsample). Using Conv+Upsample for flexibility.
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            # Node to aggregate (Sum/Concat). We use Conv to blend after addition.
            self.nodes.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, layers):
        # Project all layers to common channel dim
        projs = [proj(layer) for proj, layer in zip(self.projs, layers)]

        # Iterative aggregation:
        # Start from the deepest (last) layer
        x = projs[-1]

        # Go backwards from second to last down to first
        for i in range(len(projs) - 2, -1, -1):
            # Upsample current x
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            up = self.ups[i](up)

            # Add to the projection of the shallower level
            # Ensure spatial dims match (handle potential slight mismatch due to odd dims)
            target = projs[i]
            if up.shape[2:] != target.shape[2:]:
                up = F.interpolate(
                    up, size=target.shape[2:], mode="bilinear", align_corners=True
                )

            x = up + target
            x = self.nodes[i](x)

        return x


# ==============================================================================
# Full Model: DLASeg
# ==============================================================================


class DLASeg(nn.Module):
    def __init__(self):
        super(DLASeg, self).__init__()

        # 1. Backbone
        self.base = dla34(pretrained=False)

        # Channels for DLA-34 levels 2, 3, 4, 5 are 64, 128, 256, 512
        channels = self.base.channels
        down_ratio = 4

        # 2. Upsampling
        # We fuse levels 2, 3, 4, 5 (strides 4, 8, 16, 32)
        # Output will be at stride 4 (Level 2 resolution)
        self.dla_up = IDAUp(
            in_channels_list=[channels[2], channels[3], channels[4], channels[5]],
            out_channels=Config.HEAD_CONV,
        )

        # 3. Heads
        self.heads = nn.ModuleDict()
        for head_name, num_output in Config.HEADS.items():
            fc = nn.Sequential(
                nn.Conv2d(
                    Config.HEAD_CONV,
                    Config.HEAD_CONV,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    Config.HEAD_CONV,
                    num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            self.heads[head_name] = fc

        # 4. Initialization
        self.init_weights()

    def init_weights(self):
        # Initialize heatmap bias to -2.19 (focal loss prior)
        if "hm" in self.heads:
            self.heads["hm"][-1].bias.data.fill_(-2.19)

        # Initialize other heads
        for head in self.heads.values():
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    # Skip the last conv of hm which we just set
                    if m is self.heads["hm"][-1]:
                        continue
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        # Returns list of features at all levels [0..5]
        features = self.base(x)

        # Select levels 2, 3, 4, 5 for fusion
        # Level 2 is stride 4
        feat_list = [features[2], features[3], features[4], features[5]]

        # Upsample and Fuse
        x = self.dla_up(feat_list)

        # Heads
        ret = {}
        for head_name, head_layer in self.heads.items():
            out = head_layer(x)

            # Geometric Contract: Explicit Interpolation
            # The network output (stride 4) must be upsampled to input resolution (stride 1)
            # to match the target grid size (250x250).
            if out.shape[2:] != Config.INPUT_SIZE:
                out = F.interpolate(
                    out, size=Config.INPUT_SIZE, mode="bilinear", align_corners=True
                )

            ret[head_name] = out

        return ret
