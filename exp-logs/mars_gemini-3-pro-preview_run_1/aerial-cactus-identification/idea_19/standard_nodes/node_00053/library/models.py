import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# Shared Components
# =========================================================================


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates feature maps based on external conditioning (file size).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Project scalar condition to 2 * channels (gamma and beta)
        self.mlp = nn.Sequential(
            nn.Linear(1, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 2 * channels),
        )

    def forward(self, x, fsize):
        # fsize: (B, 1)
        params = self.mlp(fsize)  # (B, 2*C)
        gamma, beta = torch.split(params, self.channels, dim=1)

        # Reshape for broadcasting (B, C, 1, 1)
        if x.dim() == 4:
            gamma = gamma.view(-1, self.channels, 1, 1)
            beta = beta.view(-1, self.channels, 1, 1)

        return x * (1.0 + gamma) + beta


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x)


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.fc(x)


# =========================================================================
# RepVGG Components (Structural Specialist)
# =========================================================================


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
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.activation = nn.ReLU()
        self.film = FiLM(out_channels)

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

    def forward(self, inputs, fsize):
        if self.deploy:
            x = self.rbr_reparam(inputs)
        else:
            x = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
            if self.rbr_identity is not None:
                x = x + self.rbr_identity(inputs)

        x = self.film(x, fsize)
        return self.activation(x)

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
                input_dim = self.in_channels // 1
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
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class CactusRepVGG_L(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Structural Specialist: Input 2 channels (L + Laplacian)
        in_channels = 2

        # Stem: Stride 1 to preserve 32x32
        self.stem = RepVGGBlock(in_channels, 64, stride=1)

        # Stage 1: 32x32
        self.stage1 = nn.ModuleList(
            [RepVGGBlock(64, 64, stride=1), RepVGGBlock(64, 64, stride=1)]
        )

        # Stage 2: 16x16
        self.stage2 = nn.ModuleList(
            [RepVGGBlock(64, 128, stride=2), RepVGGBlock(128, 128, stride=1)]
        )

        # Stage 3: 8x8
        self.stage3 = nn.ModuleList(
            [RepVGGBlock(128, 256, stride=2), RepVGGBlock(256, 256, stride=1)]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = ClassificationHead(256, num_classes)
        self.aux_head = AuxiliaryHead(256)

    def forward(self, x, fsize):
        x = self.stem(x, fsize)

        for block in self.stage1:
            x = block(x, fsize)
        for block in self.stage2:
            x = block(x, fsize)
        for block in self.stage3:
            x = block(x, fsize)

        x = self.gap(x).flatten(1)
        logits = self.head(x)
        quality = self.aux_head(x)
        return logits, quality

    def reparameterize(self):
        self.stem.switch_to_deploy()
        for block in self.stage1:
            block.switch_to_deploy()
        for block in self.stage2:
            block.switch_to_deploy()
        for block in self.stage3:
            block.switch_to_deploy()


# =========================================================================
# ResNet Components (Chromatic Specialist)
# =========================================================================


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.film = FiLM(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, fsize):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Modulate residual branch
        out = self.film(out, fsize)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class CactusResNet_AB(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Chromatic Specialist: Input 2 channels (A + B)
        in_channels = 2

        # Stem: Stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32
        self.stage1 = nn.ModuleList(
            [ResNetBlock(64, 64, stride=1), ResNetBlock(64, 64, stride=1)]
        )

        # Stage 2: 16x16
        self.stage2 = nn.ModuleList(
            [ResNetBlock(64, 128, stride=2), ResNetBlock(128, 128, stride=1)]
        )

        # Stage 3: 8x8
        self.stage3 = nn.ModuleList(
            [ResNetBlock(128, 256, stride=2), ResNetBlock(256, 256, stride=1)]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = ClassificationHead(256, num_classes)
        self.aux_head = AuxiliaryHead(256)

    def forward(self, x, fsize):
        x = self.stem(x)

        for block in self.stage1:
            x = block(x, fsize)
        for block in self.stage2:
            x = block(x, fsize)
        for block in self.stage3:
            x = block(x, fsize)

        x = self.gap(x).flatten(1)
        logits = self.head(x)
        quality = self.aux_head(x)
        return logits, quality


# =========================================================================
# NeXt Components (Holistic Specialist)
# =========================================================================


class NeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )  # Depthwise
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.film = FiLM(dim)

    def forward(self, x, fsize):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        # Apply FiLM
        x = x.permute(0, 3, 1, 2)  # Back to (N, C, H, W)
        x = self.film(x, fsize)
        x = x.permute(0, 2, 3, 1)  # To (N, H, W, C)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class NeXtDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # N C H W -> N H W C
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # N H W C -> N C H W
        x = self.conv(x)
        return x


class CactusNeXt_RGB(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Holistic Specialist: Input 3 channels (RGB)
        in_channels = 3

        # Stem: 3x3 Conv stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 32, 32], eps=1e-6),  # LayerNorm over C, H, W? Or just C?
            # ConvNeXt stem usually: Conv -> LN. LN in PyTorch expects last dims.
            # If input is NCHW, LN([C,H,W]) works.
        )

        # Stage 1: 32x32
        self.stage1 = nn.ModuleList([NeXtBlock(64), NeXtBlock(64)])

        # Downsample 1
        self.down1 = NeXtDownsample(64, 128)

        # Stage 2: 16x16
        self.stage2 = nn.ModuleList([NeXtBlock(128), NeXtBlock(128)])

        # Downsample 2
        self.down2 = NeXtDownsample(128, 256)

        # Stage 3: 8x8
        self.stage3 = nn.ModuleList([NeXtBlock(256), NeXtBlock(256)])

        self.norm_final = nn.LayerNorm(256, eps=1e-6)
        self.head = ClassificationHead(256, num_classes)
        self.aux_head = AuxiliaryHead(256)

    def forward(self, x, fsize):
        x = self.stem(x)

        for block in self.stage1:
            x = block(x, fsize)

        x = self.down1(x)

        for block in self.stage2:
            x = block(x, fsize)

        x = self.down2(x)

        for block in self.stage3:
            x = block(x, fsize)

        # Global Pooling
        x = x.mean([-2, -1])  # Global Average Pooling over H, W
        x = self.norm_final(x)

        logits = self.head(x)
        quality = self.aux_head(x)
        return logits, quality
