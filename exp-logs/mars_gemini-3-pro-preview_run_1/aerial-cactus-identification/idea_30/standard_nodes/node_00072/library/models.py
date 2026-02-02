import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
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
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.non_linearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.groups,
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
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding - kernel_size // 2,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
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
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
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
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            groups=self.rbr_dense.conv.groups,
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


class CactusRepVGG(nn.Module):
    def __init__(self, num_classes=1, width_multiplier=None):
        super(CactusRepVGG, self).__init__()

        # Architecture configuration for 32x32 input
        # Stage 0: Stem (32x32)
        # Stage 1: 32x32
        # Stage 2: 16x16
        # Stage 3: 8x8
        # Stage 4: 4x4

        self.stages_config = [64, 64, 128, 256, 512]
        if width_multiplier:
            self.stages_config = [int(x * width_multiplier) for x in self.stages_config]

        self.in_planes = min(64, self.stages_config[0])

        # Stage 0: Stem - Conservative downsampling (Stride 1, No pooling)
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=False,
        )
        self.cur_layer_idx = 1

        # Stage 1
        self.stage1 = self._make_stage(self.stages_config[1], num_blocks=2, stride=1)
        # Stage 2
        self.stage2 = self._make_stage(self.stages_config[2], num_blocks=2, stride=2)
        # Stage 3
        self.stage3 = self._make_stage(self.stages_config[3], num_blocks=2, stride=2)
        # Stage 4
        self.stage4 = self._make_stage(self.stages_config[4], num_blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Multi-Task Heads
        self.classifier = nn.Linear(self.stages_config[4], num_classes)
        self.aux_head = nn.Linear(self.stages_config[4], 1)  # Predicts log file size

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=s,
                    padding=1,
                    deploy=False,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

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

        return class_logits, quality_pred

    def reparameterize(self):
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()


# ---------------------------------------------------------------------------------
# ResNet Implementation
# ---------------------------------------------------------------------------------


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
            # Using 1x1 for projection shortcut is standard ResNet
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
    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Conservative Stem: 3x3 conv, stride 1, no pooling to preserve 32x32 resolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layers
        # Layer 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # Layer 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # Layer 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # Layer 4: 8x8 -> 4x4
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-Task Heads
        self.classifier = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.aux_head = nn.Linear(512 * BasicBlock.expansion, 1)

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
        out = self.layer4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        class_logits = self.classifier(out)
        quality_pred = self.aux_head(out)

        return class_logits, quality_pred
