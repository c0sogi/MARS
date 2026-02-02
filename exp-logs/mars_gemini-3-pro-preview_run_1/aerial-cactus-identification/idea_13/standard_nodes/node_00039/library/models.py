import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config

# -----------------------------------------------------------------------------
# Metadata Gating Module
# -----------------------------------------------------------------------------


class MetadataGate(nn.Module):
    """
    A lightweight MLP that learns to gate image features based on metadata (file size).
    """

    def __init__(self, input_dim=1, hidden_dim=16, output_dim=64):
        super(MetadataGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, features, metadata):
        """
        Args:
            features: (B, C) Image feature vector.
            metadata: (B,) or (B, 1) Normalized file size.
        Returns:
            Gated features (B, C).
        """
        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(1)

        # Generate gate vector (0 to 1)
        gate = self.mlp(metadata)

        # Element-wise multiplication
        return features * gate


# -----------------------------------------------------------------------------
# RepVGG Components
# -----------------------------------------------------------------------------


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
        self.activation = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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
                padding=0,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.activation(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

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


# -----------------------------------------------------------------------------
# Model 1: CactusRepVGG (Gated)
# -----------------------------------------------------------------------------


class CactusRepVGG(nn.Module):
    def __init__(self, num_classes=1, deploy=False):
        super(CactusRepVGG, self).__init__()

        # Conservative stem: 3x3, stride 1, no pooling to preserve 32x32
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        # Stages with downsampling via stride=2
        self.stage1 = self._make_stage(32, 64, 2, stride=2, deploy=deploy)  # 16x16
        self.stage2 = self._make_stage(64, 128, 2, stride=2, deploy=deploy)  # 8x8
        self.stage3 = self._make_stage(128, 256, 2, stride=2, deploy=deploy)  # 4x4

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Metadata Gating
        self.gate = MetadataGate(input_dim=1, hidden_dim=32, output_dim=256)

        self.linear = nn.Linear(256, num_classes)

    def _make_stage(self, in_planes, planes, num_blocks, stride, deploy):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=s,
                    padding=1,
                    deploy=deploy,
                )
            )
            in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x, metadata):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        # Apply Gating
        out = self.gate(out, metadata)

        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


# -----------------------------------------------------------------------------
# Model 2: CactusResNet (Gated)
# -----------------------------------------------------------------------------


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
    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()
        self.in_planes = 32

        # Conservative Stem: No MaxPool, Stride 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 2, stride=1)  # 32x32
        self.layer2 = self._make_layer(64, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(256, 2, stride=2)  # 4x4

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Metadata Gating
        self.gate = MetadataGate(input_dim=1, hidden_dim=32, output_dim=256)

        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, metadata):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        out = self.gate(out, metadata)

        out = self.linear(out)
        return out


# -----------------------------------------------------------------------------
# Model 3: CactusNeXt (Gated, 3x3 Only)
# -----------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class NeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # Depthwise conv 3x3 (strictly 3x3 for small images)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise convs (1x1) implemented as Linear
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class CactusNeXt(nn.Module):
    def __init__(self, num_classes=1, depths=[2, 2, 2, 2], dims=[32, 64, 128, 256]):
        super(CactusNeXt, self).__init__()

        # Conservative Stem: Patchify with stride 1 (effectively just a conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Stage 0 (32x32) - No downsample before this
        self.downsample_layers.append(nn.Identity())

        # Downsample layers for subsequent stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        for i in range(4):
            stage = nn.Sequential(*[NeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = LayerNorm(
            dims[-1], eps=1e-6, data_format="channels_first"
        )  # Final norm

        # Metadata Gating
        self.gate = MetadataGate(input_dim=1, hidden_dim=32, output_dim=dims[-1])

        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x, metadata):
        x = self.stem(x)

        # Stage 0
        x = self.stages[0](x)

        # Stage 1
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        # Stage 2
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)

        # Stage 3
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)

        x = self.norm(x)
        x = x.mean([-2, -1])  # Global Average Pooling

        # Gating
        x = self.gate(x, metadata)

        x = self.head(x)
        return x
