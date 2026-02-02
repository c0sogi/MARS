import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import NUM_CLASSES


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """Helper for Conv + BN sequence."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """
    RepVGG Block:
    Training: 3x3 Conv + 1x1 Conv + Identity (if applicable)
    Inference: Fused 3x3 Conv
    """

    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.non_linearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

    def forward(self, inputs):
        if self.deploy:
            return self.non_linearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Identity branch is just BatchNorm
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // 1  # groups=1
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=3,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
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
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class AuxHead(nn.Module):
    """Auxiliary Classification Head for Deep Supervision."""

    def __init__(self, in_channels, num_classes):
        super(AuxHead, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class CactusRepVGG_DS(nn.Module):
    """
    Custom RepVGG for 32x32 images with Deep Supervision.
    Structure:
    - Stem (32x32)
    - Stage 1 (16x16)
    - Stage 2 (8x8) -> Aux Head
    - Stage 3 (4x4)
    - Global Pool + FC
    """

    def __init__(self, num_classes=NUM_CLASSES, deploy=False):
        super(CactusRepVGG_DS, self).__init__()
        self.deploy = deploy

        # Hyperparameters for channel widths
        self.stages = [64, 128, 256, 512]

        # Stem: 3x3 Conv, Stride 1 (Conservative Downsampling)
        self.stage0 = RepVGGBlock(3, self.stages[0], stride=1, deploy=deploy)

        # Stage 1: 32x32 -> 16x16
        self.stage1 = self._make_stage(
            self.stages[0], self.stages[1], num_blocks=2, stride=2, deploy=deploy
        )

        # Stage 2: 16x16 -> 8x8
        self.stage2 = self._make_stage(
            self.stages[1], self.stages[2], num_blocks=2, stride=2, deploy=deploy
        )

        # Auxiliary Head attached after Stage 2
        self.aux_head = AuxHead(self.stages[2], num_classes)

        # Stage 3: 8x8 -> 4x4
        self.stage3 = self._make_stage(
            self.stages[2], self.stages[3], num_blocks=2, stride=2, deploy=deploy
        )

        # Final Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.stages[3], num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, deploy):
        layers = []
        layers.append(
            RepVGGBlock(in_channels, out_channels, stride=stride, deploy=deploy)
        )
        for _ in range(1, num_blocks):
            layers.append(
                RepVGGBlock(out_channels, out_channels, stride=1, deploy=deploy)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        # Deep Supervision
        aux_out = None
        if self.training:
            aux_out = self.aux_head(x)

        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        main_out = self.linear(x)

        if self.training:
            return main_out, aux_out
        return main_out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


# ==========================================================================================
# ResNet Implementation
# ==========================================================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CactusResNet_DS(nn.Module):
    """
    ResNet adapted for 32x32 images with Deep Supervision.
    Structure:
    - Stem (32x32, 3x3 Conv, No MaxPool)
    - Layer 1 (32x32)
    - Layer 2 (16x16)
    - Layer 3 (8x8) -> Aux Head
    - Layer 4 (4x4)
    - Global Pool + FC
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(CactusResNet_DS, self).__init__()
        self.in_planes = 64

        # Conservative Stem: 3x3 conv, stride 1, padding 1. No MaxPool.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        # Auxiliary Head after Layer 3 (8x8 feature map)
        self.aux_head = AuxHead(256 * BasicBlock.expansion, num_classes)

        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Deep Supervision
        aux_out = None
        if self.training:
            aux_out = self.aux_head(out)

        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        main_out = self.linear(out)

        if self.training:
            return main_out, aux_out
        return main_out
