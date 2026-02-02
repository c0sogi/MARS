import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import NUM_CLASSES

# =============================================================================
# SHARED COMPONENTS
# =============================================================================


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# CACTUS REPVGG
# =============================================================================


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        # Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # Pad 1x1 kernel to 3x3
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)
        # Fuse identity branch
        kernelid, biasid = self._fuse_identity()

        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch[0].weight
        running_mean = branch[1].running_mean
        running_var = branch[1].running_var
        gamma = branch[1].weight
        beta = branch[1].bias
        eps = branch[1].eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_identity(self):
        if self.rbr_identity is None:
            return 0, 0
        kernel_value = torch.zeros(self.in_channels, self.in_channels, 3, 3)
        for i in range(self.in_channels):
            kernel_value[i, i, 1, 1] = 1
        kernel_value = kernel_value.to(self.rbr_identity.weight.device)

        running_mean = self.rbr_identity.running_mean
        running_var = self.rbr_identity.running_var
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        eps = self.rbr_identity.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel_value * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        self.deploy = True


class CactusRepVGG(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CactusRepVGG, self).__init__()

        # Conservative Downsampling Stem: 32x32 -> 32x32
        self.stem = RepVGGBlock(3, 64, stride=1)

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(
            RepVGGBlock(64, 64, stride=1), RepVGGBlock(64, 64, stride=1)
        )

        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            RepVGGBlock(64, 128, stride=2), RepVGGBlock(128, 128, stride=1)
        )

        # Aux Head attached after Stage 2 (16x16)
        self.aux_head = AuxiliaryHead(128, num_classes)

        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            RepVGGBlock(128, 256, stride=2), RepVGGBlock(256, 256, stride=1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)

        if self.training:
            aux = self.aux_head(x)

        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        main = self.fc(x)

        if self.training:
            return main, aux
        return main

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


# =============================================================================
# CACTUS RESNET
# =============================================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CactusResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Conservative Downsampling Stem: 32x32 -> 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Layer 1: 32x32
        self.layer1 = self._make_layer(64, 2, stride=1)

        # Layer 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(128, 2, stride=2)

        # Aux Head
        self.aux_head = AuxiliaryHead(128, num_classes)

        # Layer 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        if self.training:
            aux = self.aux_head(out)

        out = self.layer3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        main = self.fc(out)

        if self.training:
            return main, aux
        return main


# =============================================================================
# CACTUS MICRONEXT
# =============================================================================


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # Depthwise Conv 3x3 (Small Kernel Adaptation)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm2d(dim)
        # Inverted Bottleneck: Expand -> GELU -> Shrink
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()  # Simplified for this scale

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Permute for linear layers: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class CactusMicroNeXt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CactusMicroNeXt, self).__init__()

        dims = [64, 128, 256]

        # Conservative Downsampling Stem: 3x3 patchify, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0]),
        )

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(NeXtBlock(dims[0]), NeXtBlock(dims[0]))

        # Downsample 1: 32x32 -> 16x16
        self.downsample1 = nn.Sequential(
            LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )

        # Stage 2: 16x16
        self.stage2 = nn.Sequential(NeXtBlock(dims[1]), NeXtBlock(dims[1]))

        # Aux Head
        self.aux_head = AuxiliaryHead(dims[1], num_classes)

        # Downsample 2: 16x16 -> 8x8
        self.downsample2 = nn.Sequential(
            LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )

        # Stage 3: 8x8
        self.stage3 = nn.Sequential(NeXtBlock(dims[2]), NeXtBlock(dims[2]))

        self.norm = LayerNorm2d(dims[2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dims[2], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)

        if self.training:
            aux = self.aux_head(x)

        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.norm(x)
        x = self.gap(x)
        x = x.flatten(1)
        main = self.head(x)

        if self.training:
            return main, aux
        return main


# =============================================================================
# MODEL FACTORY
# =============================================================================


class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes=NUM_CLASSES):
        if model_name == "RepVGG":
            return CactusRepVGG(num_classes=num_classes)
        elif model_name == "ResNet":
            return CactusResNet(num_classes=num_classes)
        elif model_name == "NeXt":
            return CactusMicroNeXt(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")
