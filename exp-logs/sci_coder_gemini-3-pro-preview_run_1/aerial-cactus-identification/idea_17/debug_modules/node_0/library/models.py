import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


# ------------------------------------------------------------------------------
# 1. Feature-wise Linear Modulation (FiLM)
# ------------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    Projects scalar metadata (file size) into scale (gamma) and shift (beta)
    parameters to modulate feature maps.
    """

    def __init__(self, channels, metadata_dim=1):
        super(FiLMLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(metadata_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2 * channels),
        )

    def forward(self, x, film_input):
        # x: (B, C, H, W)
        # film_input: (B, 1) or (B,)

        if film_input.dim() == 1:
            film_input = film_input.unsqueeze(1)

        # Project metadata to style parameters
        style = self.fc(film_input)  # (B, 2*C)
        gamma, beta = torch.chunk(style, 2, dim=1)

        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        # Modulate: Out = (1 + gamma) * In + beta
        return x * (1.0 + gamma) + beta


# ------------------------------------------------------------------------------
# 2. Building Blocks
# ------------------------------------------------------------------------------
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
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # FiLM Layer
        self.film = FiLMLayer(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if deploy:
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
            self.rbr_dense = conv_bn(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.rbr_1x1 = conv_bn(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x, film_input):
        if self.deploy:
            out = self.rbr_reparam(x)
        else:
            out = self.rbr_dense(x) + self.rbr_1x1(x)
            if self.rbr_identity is not None:
                out = out + self.rbr_identity(x)

        # Apply FiLM before activation
        out = self.film(out, film_input)
        return self.activation(out)

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

        if hasattr(self, "rbr_dense"):
            del self.rbr_dense
        if hasattr(self, "rbr_1x1"):
            del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity

        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        kernelid, biasid = 0, 0
        if self.rbr_identity is not None:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is 0:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

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
            # Identity BN case
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i, 1, 1] = 1
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

        self.film = FiLMLayer(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, film_input):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply FiLM before residual addition
        out = self.film(out, film_input)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(NeXtBlock, self).__init__()

        # 1. Depthwise Conv (Strictly 3x3)
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            stride=stride,
        )

        # 2. LayerNorm
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # 3. FiLM (Applied after Norm)
        self.film = FiLMLayer(in_channels)

        # 4. Pointwise MLP
        # Handle channel expansion if needed (though typically NeXt keeps dims inside block)
        # Here we follow ConvNeXt style: 4x expansion in MLP
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)

        # Projection for residual if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x, film_input):
        input = x

        x = self.dwconv(x)

        # Permute for LN and MLP: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # Permute back for FiLM: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.film(x, film_input)

        # Permute again for MLP
        x = x.permute(0, 2, 3, 1)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Permute back
        x = x.permute(0, 3, 1, 2)

        x = input + x if input.shape == x.shape else self.shortcut(input) + x
        return x


# ------------------------------------------------------------------------------
# 3. Backbones
# ------------------------------------------------------------------------------
class CactusRepVGG(nn.Module):
    def __init__(self, deploy=False):
        super(CactusRepVGG, self).__init__()
        self.deploy = deploy

        # Stem: 3x3, Stride 1 (Preserve 32x32)
        self.stem = RepVGGBlock(3, 64, stride=1, deploy=deploy)

        # Stages
        self.stage1 = self._make_layer(64, 64, 3)
        self.stage2 = self._make_layer(64, 128, 3)
        self.stage3 = self._make_layer(128, 256, 3)

        self.out_channels = 256

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = nn.ModuleList()
        layers.append(
            RepVGGBlock(in_channels, out_channels, stride=1, deploy=self.deploy)
        )
        for _ in range(1, blocks):
            layers.append(
                RepVGGBlock(out_channels, out_channels, stride=1, deploy=self.deploy)
            )
        return layers

    def forward(self, x, film_input):
        x = self.stem(x, film_input)
        for layer in self.stage1:
            x = layer(x, film_input)
        for layer in self.stage2:
            x = layer(x, film_input)
        for layer in self.stage3:
            x = layer(x, film_input)
        return x


class CactusResNet(nn.Module):
    def __init__(self):
        super(CactusResNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_layer(64, 64, 3)
        self.stage2 = self._make_layer(64, 128, 3)
        self.stage3 = self._make_layer(128, 256, 3)

        self.out_channels = 256

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = nn.ModuleList()
        layers.append(ResNetBlock(in_channels, out_channels, stride=1))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return layers

    def forward(self, x, film_input):
        x = self.stem(x)
        for layer in self.stage1:
            x = layer(x, film_input)
        for layer in self.stage2:
            x = layer(x, film_input)
        for layer in self.stage3:
            x = layer(x, film_input)
        return x


class CactusNeXt(nn.Module):
    def __init__(self):
        super(CactusNeXt, self).__init__()

        # Stem: Patchify (3x3 conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 32, 32], eps=1e-6),
        )

        self.stage1 = self._make_layer(64, 64, 3)
        self.stage2 = self._make_layer(64, 128, 3)
        self.stage3 = self._make_layer(128, 256, 3)

        self.out_channels = 256

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = nn.ModuleList()
        layers.append(NeXtBlock(in_channels, out_channels, stride=1))
        for _ in range(1, blocks):
            layers.append(NeXtBlock(out_channels, out_channels, stride=1))
        return layers

    def forward(self, x, film_input):
        x = self.stem(x)
        for layer in self.stage1:
            x = layer(x, film_input)
        for layer in self.stage2:
            x = layer(x, film_input)
        for layer in self.stage3:
            x = layer(x, film_input)
        return x


# ------------------------------------------------------------------------------
# 4. Main Model Wrapper
# ------------------------------------------------------------------------------
class CactusModel(nn.Module):
    def __init__(self, backbone_name="RepVGG", num_classes=1):
        super(CactusModel, self).__init__()

        # Select Backbone
        if backbone_name == "RepVGG":
            self.backbone = CactusRepVGG(deploy=False)
        elif backbone_name == "ResNet":
            self.backbone = CactusResNet()
        elif backbone_name == "NeXt":
            self.backbone = CactusNeXt()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone_name = backbone_name
        feat_dim = self.backbone.out_channels

        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Heads
        self.head = nn.Linear(feat_dim, num_classes)
        self.aux_head = nn.Linear(feat_dim, 1)  # Predicts log(file_size)

    def forward(self, x, film_input):
        # Extract features
        features = self.backbone(x, film_input)

        # Pooling
        pooled = self.pool(features).flatten(1)

        # Predictions
        logits = self.head(pooled)
        quality_pred = self.aux_head(pooled)

        return logits, quality_pred

    def switch_to_deploy(self):
        """
        Fuses RepVGG blocks for faster inference.
        """
        if self.backbone_name == "RepVGG":
            for module in self.backbone.modules():
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()
