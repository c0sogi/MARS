import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import Config

# --------------------------------------------------------------------------
# RepVGG Utilities and Blocks
# --------------------------------------------------------------------------


def fuse_bn_tensor(conv, bn):
    """
    Fuses the weights and bias of a convolution layer with its following batch normalization.
    Returns the fused kernel and bias.
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)

    return kernel * t, beta - running_mean * gamma / std


class RepVGGBlock(nn.Module):
    """
    RepVGG Block:
    Training: 3x3 Branch + 1x1 Branch + Identity Branch (if applicable).
    Inference: Fused into a single 3x3 Convolution.
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.activation = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            # 3x3 Branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # 1x1 Branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # Identity Branch (only if dimensions match)
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(out_channels)
            else:
                self.rbr_identity = None

    def forward(self, inputs):
        if self.deploy:
            return self.activation(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = fuse_bn_tensor(self.rbr_dense[0], self.rbr_dense[1])

        # Fuse 1x1 branch
        kernel1x1, bias1x1 = fuse_bn_tensor(self.rbr_1x1[0], self.rbr_1x1[1])

        # Fuse Identity branch
        kernelid, biasid = 0, 0
        if self.rbr_identity is not None:
            # Create a pseudo 1x1 identity convolution
            running_mean = self.rbr_identity.running_mean
            running_var = self.rbr_identity.running_var
            gamma = self.rbr_identity.weight
            beta = self.rbr_identity.bias
            eps = self.rbr_identity.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)

            # Identity kernel is 1 at pos (c, c) and 0 elsewhere
            input_dim = self.in_channels
            kernel_value = torch.zeros(self.in_channels, input_dim, 3, 3).to(
                self.rbr_dense[0].weight.device
            )
            for i in range(self.in_channels):
                kernel_value[i, i, 1, 1] = 1

            kernelid = kernel_value * t
            biasid = beta - running_mean * gamma / std

        # Pad 1x1 kernel to 3x3
        # 1x1 kernel is centered, so we pad 1 pixel on all sides
        kernel1x1_padded = F.pad(kernel1x1, [1, 1, 1, 1])

        return kernel3x3 + kernel1x1_padded + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class CactusRepVGG(nn.Module):
    """
    RepVGG Architecture adapted for 32x32 input.
    Features:
    - Conservative Stem (3x3, stride 1, no pool)
    - 4 Stages of RepVGG Blocks
    - Auxiliary Quality Head
    """

    def __init__(self, num_classes=1, deploy=False):
        super(CactusRepVGG, self).__init__()
        self.deploy = deploy

        # Channel configurations for stages
        # Input: 32x32
        self.stage0 = RepVGGBlock(
            in_channels=3, out_channels=64, stride=1, padding=1, deploy=deploy
        )

        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1, deploy=deploy)
        self.stage2 = self._make_stage(
            64, 128, num_blocks=2, stride=2, deploy=deploy
        )  # 16x16
        self.stage3 = self._make_stage(
            128, 256, num_blocks=2, stride=2, deploy=deploy
        )  # 8x8
        self.stage4 = self._make_stage(
            256, 512, num_blocks=2, stride=2, deploy=deploy
        )  # 4x4

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Heads
        self.classifier = nn.Linear(512, num_classes)
        self.aux_head = nn.Linear(512, 1)  # Predicts log file size

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, deploy):
        layers = []
        layers.append(
            RepVGGBlock(
                in_channels, out_channels, stride=stride, padding=1, deploy=deploy
            )
        )
        for _ in range(num_blocks - 1):
            layers.append(
                RepVGGBlock(
                    out_channels, out_channels, stride=1, padding=1, deploy=deploy
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        class_logits = self.classifier(out)
        quality_pred = self.aux_head(out)

        # During training/val we might need both.
        # The loss function will handle the dictionary.
        return {"class": class_logits, "quality": quality_pred}

    def switch_to_deploy(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True


# --------------------------------------------------------------------------
# ResNet Components
# --------------------------------------------------------------------------


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


class CactusResNet(nn.Module):
    """
    ResNet Architecture adapted for 32x32 input.
    Features:
    - Conservative Stem (3x3, stride 1, no pool)
    - Standard Residual Blocks
    - Auxiliary Quality Head
    """

    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Stem: 3x3 conv, stride 1, no pooling (preserves 32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)  # 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 4x4

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Heads
        self.classifier = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.aux_head = nn.Linear(512 * BasicBlock.expansion, 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        class_logits = self.classifier(out)
        quality_pred = self.aux_head(out)

        return {"class": class_logits, "quality": quality_pred}
