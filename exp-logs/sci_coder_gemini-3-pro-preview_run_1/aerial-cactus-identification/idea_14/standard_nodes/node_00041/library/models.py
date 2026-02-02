import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# -------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation) Components
# -------------------------------------------------------------------------


class FiLMGenerator(nn.Module):
    """
    Generates scale (gamma) and shift (beta) parameters from metadata (file size).
    """

    def __init__(self, input_dim=1, num_features=64, hidden_dim=32):
        super(FiLMGenerator, self).__init__()
        self.num_features = num_features
        # Simple MLP: Input -> Linear -> ReLU -> Linear -> (Gamma, Beta)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_features),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize last layer to produce gamma=0, beta=0 initially
        # This ensures the modulation starts as identity: (1+0)*x + 0 = x
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x_meta):
        # x_meta: (B, 1)
        out = self.mlp(x_meta)  # (B, 2*C)
        gamma, beta = torch.split(out, self.num_features, dim=1)
        # Reshape for broadcasting over spatial dims: (B, C, 1, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation.
    """

    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()
        self.generator = FiLMGenerator(num_features=num_features)

    def forward(self, x, x_meta):
        gamma, beta = self.generator(x_meta)
        # Apply modulation: (1 + gamma) * x + beta
        return (1 + gamma) * x + beta


# -------------------------------------------------------------------------
# 1. CactusRepVGG-FiLM
# -------------------------------------------------------------------------


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
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels  # Add this line to store out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

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
                padding_mode=padding_mode,
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
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

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


class CactusRepVGG(nn.Module):
    def __init__(self, num_classes=1, width_multiplier=1.0, deploy=False):
        super(CactusRepVGG, self).__init__()

        self.deploy = deploy

        # Configurations for 3 stages
        # 32x32 input.
        # Stage 0: Stem (stride 1) -> 32x32
        # Stage 1: Stride 2 -> 16x16
        # Stage 2: Stride 2 -> 8x8
        # Stage 3: Stride 2 -> 4x4 (Optional, or just stay at 8x8. Let's do 3 stages total after stem)

        planes = [
            int(64 * width_multiplier),
            int(128 * width_multiplier),
            int(256 * width_multiplier),
        ]
        self.planes = planes

        # Stem
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=planes[0],
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )
        self.film0 = FiLMLayer(planes[0])

        # Stage 1
        self.stage1 = self._make_stage(
            planes[0], planes[1], num_blocks=2, stride=2, deploy=deploy
        )
        self.film1 = FiLMLayer(planes[1])

        # Stage 2
        self.stage2 = self._make_stage(
            planes[1], planes[2], num_blocks=2, stride=2, deploy=deploy
        )
        self.film2 = FiLMLayer(planes[2])

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(planes[2], num_classes)

    def _make_stage(self, in_planes, out_planes, num_blocks, stride, deploy):
        layers = []
        layers.append(
            RepVGGBlock(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                deploy=deploy,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                RepVGGBlock(
                    out_planes,
                    out_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=deploy,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x, fsize):
        # fsize is (N, 1)

        # Stage 0
        out = self.stage0(x)
        out = self.film0(out, fsize)

        # Stage 1
        out = self.stage1(out)
        out = self.film1(out, fsize)

        # Stage 2
        out = self.stage2(out)
        out = self.film2(out, fsize)

        # Head
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.stage0.switch_to_deploy()
        for layer in self.stage1:
            layer.switch_to_deploy()
        for layer in self.stage2:
            layer.switch_to_deploy()
        self.deploy = True


# -------------------------------------------------------------------------
# 2. CactusResNet-FiLM
# -------------------------------------------------------------------------


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CactusResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()

        # Stride 1 stem to preserve 32x32
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.film1 = FiLMLayer(64)

        self.layer2 = self._make_layer(128, 2, stride=2)
        self.film2 = FiLMLayer(128)

        self.layer3 = self._make_layer(256, 2, stride=2)
        self.film3 = FiLMLayer(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, fsize):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.film1(out, fsize)

        out = self.layer2(out)
        out = self.film2(out, fsize)

        out = self.layer3(out)
        out = self.film3(out, fsize)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# -------------------------------------------------------------------------
# 3. CactusNeXt-FiLM
# -------------------------------------------------------------------------


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
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()  # Placeholder for simplicity

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class CactusNeXt(nn.Module):
    def __init__(self, num_classes=1, depths=[2, 2, 2], dims=[64, 128, 256]):
        super(CactusNeXt, self).__init__()

        # Stem: 3x3 conv, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # Stages
        self.stages = nn.ModuleList()
        self.films = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Stage 0 (Resolution 32x32)
        self.stages.append(self._make_stage(dims[0], depths[0]))
        self.films.append(FiLMLayer(dims[0]))
        self.downsamples.append(nn.Identity())  # No downsample for first stage

        # Stage 1 (Resolution 16x16)
        # Downsample layer
        self.downsamples.append(
            nn.Sequential(
                LayerNorm2d(dims[0], eps=1e-6),
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
            )
        )
        self.stages.append(self._make_stage(dims[1], depths[1]))
        self.films.append(FiLMLayer(dims[1]))

        # Stage 2 (Resolution 8x8)
        self.downsamples.append(
            nn.Sequential(
                LayerNorm2d(dims[1], eps=1e-6),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            )
        )
        self.stages.append(self._make_stage(dims[2], depths[2]))
        self.films.append(FiLMLayer(dims[2]))

        self.norm = LayerNorm2d(dims[-1], eps=1e-6)  # Final norm
        self.head = nn.Linear(dims[-1], num_classes)

    def _make_stage(self, dim, depth):
        layers = []
        for _ in range(depth):
            layers.append(NeXtBlock(dim))
        return nn.Sequential(*layers)

    def forward(self, x, fsize):
        x = self.stem(x)

        # Loop through stages
        for i in range(len(self.stages)):
            if i > 0:
                x = self.downsamples[i](x)
            x = self.stages[i](x)
            x = self.films[i](x, fsize)

        x = self.norm(x)
        x = x.mean([-2, -1])  # Global Avg Pooling
        x = self.head(x)
        return x
