import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


# ==========================================
# 1. CactusResNet (Conservative Downsampling)
# ==========================================
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
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 64

        # Conservative Stem: 3x3 conv, stride 1 (no 7x7 stride 2)
        # Keeps 32x32 resolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # No MaxPool here for 32x32 images

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = self.layer3(out)  # 8x8
        out = self.layer4(out)  # 4x4
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==========================================
# 2. CactusRepVGG (Re-parameterizable)
# ==========================================
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
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.non_linearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
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
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,  # 1x1 conv has 0 padding
                groups=groups,
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
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=np.float32,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
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
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
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

    def _get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is 0:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


class CactusRepVGG(nn.Module):
    def __init__(self, num_classes=1, width_multiplier=1.0):
        super(CactusRepVGG, self).__init__()

        # Structure adapted for 32x32
        self.in_planes = min(64, int(64 * width_multiplier))

        # Stage 0: Stem (Stride 1)
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=False,
        )

        # Stage 1: Stride 1 (Keep resolution high for a bit)
        planes_1 = int(64 * width_multiplier)
        self.stage1 = RepVGGBlock(self.in_planes, planes_1, stride=1, deploy=False)
        self.in_planes = planes_1

        # Stage 2: Stride 2 (16x16)
        planes_2 = int(128 * width_multiplier)
        self.stage2 = RepVGGBlock(self.in_planes, planes_2, stride=2, deploy=False)
        self.in_planes = planes_2

        # Stage 3: Stride 2 (8x8)
        planes_3 = int(256 * width_multiplier)
        self.stage3 = RepVGGBlock(self.in_planes, planes_3, stride=2, deploy=False)
        self.in_planes = planes_3

        # Stage 4: Stride 2 (4x4)
        planes_4 = int(512 * width_multiplier)
        self.stage4 = RepVGGBlock(self.in_planes, planes_4, stride=2, deploy=False)
        self.in_planes = planes_4

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(planes_4, num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


# ==========================================
# 3. CactusNeXt (3x3 Kernel Adaptation)
# ==========================================
class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
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
        # Depthwise Conv 3x3 (instead of 7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm2d(dim)
        # Pointwise Convs implemented as 1x1 Conv2d
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

        self.drop_path = (
            nn.Identity()
        )  # Simplification: No stochastic depth for this scale

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class CactusNeXt(nn.Module):
    def __init__(self, num_classes=1, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512]):
        super(CactusNeXt, self).__init__()

        # Stem: 3x3 stride 1 (Conservative)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[NeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1])  # Global Avg Pooling
        x = self.norm(x)
        x = self.head(x)
        return x


# ==========================================
# Factory Function
# ==========================================
def create_model(model_name, num_classes=1, pretrained=False):
    """
    Factory function to create models by name.

    Args:
        model_name (str): 'resnet', 'repvgg', or 'next'.
        num_classes (int): Number of output classes (default 1).
        pretrained (bool): Ignored, as these are custom architectures.

    Returns:
        nn.Module: The requested model.
    """
    model_name = model_name.lower()

    if model_name == "resnet":
        return CactusResNet(num_classes=num_classes)

    elif model_name == "repvgg":
        return CactusRepVGG(num_classes=num_classes)

    elif model_name == "next":
        return CactusNeXt(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from ['resnet', 'repvgg', 'next']"
        )
