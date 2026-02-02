import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# Shared Components
# -----------------------------------------------------------------------------


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates feature maps based on a conditioning input (file size).
    Formula: out = (1 + gamma(z)) * in + beta(z)
    """

    def __init__(self, in_channels, cond_dim=1):
        super().__init__()
        # Project conditioning variable to 2 * in_channels (gamma and beta)
        self.fc = nn.Linear(cond_dim, 2 * in_channels)

        # Initialize to identity: gamma=0 (so 1+gamma=1), beta=0
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, cond):
        # x: (B, C, H, W)
        # cond: (B,) or (B, 1)
        if cond.dim() == 1:
            cond = cond.unsqueeze(1)

        params = self.fc(cond)  # (B, 2*C)
        gamma, beta = torch.split(params, x.shape[1], dim=1)

        # Reshape for broadcasting to spatial dimensions
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        beta = beta.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)

        return (1.0 + gamma) * x + beta


# -----------------------------------------------------------------------------
# 1. CactusRepVGG_MTL
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
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.non_linearity = nn.ReLU()

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

    def forward(self, inputs):
        if hasattr(self, "deploy") and self.deploy:
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

    def switch_to_deploy(self):
        if hasattr(self, "deploy") and self.deploy:
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
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class CactusRepVGG_MTL(nn.Module):
    def __init__(self, num_classes=1, widths=[64, 128, 256], deploy=False):
        super().__init__()
        self.deploy = deploy

        # Stem: Stride 1, 32x32 -> 32x32 (Conservative Downsampling)
        self.stem = RepVGGBlock(3, widths[0], stride=1, deploy=deploy)
        self.film0 = FiLMLayer(widths[0])

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(
            RepVGGBlock(widths[0], widths[0], stride=1, deploy=deploy),
            RepVGGBlock(widths[0], widths[0], stride=1, deploy=deploy),
        )
        self.film1 = FiLMLayer(widths[0])

        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            RepVGGBlock(widths[0], widths[1], stride=2, deploy=deploy),
            RepVGGBlock(widths[1], widths[1], stride=1, deploy=deploy),
        )
        self.film2 = FiLMLayer(widths[1])

        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            RepVGGBlock(widths[1], widths[2], stride=2, deploy=deploy),
            RepVGGBlock(widths[2], widths[2], stride=1, deploy=deploy),
        )
        self.film3 = FiLMLayer(widths[2])

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Multi-Task Heads
        self.head_class = nn.Linear(widths[2], num_classes)
        self.head_aux = nn.Linear(widths[2], 1)  # Predicts log file size

    def forward(self, x, film_feat):
        x = self.stem(x)
        x = self.film0(x, film_feat)

        x = self.stage1(x)
        x = self.film1(x, film_feat)

        x = self.stage2(x)
        x = self.film2(x, film_feat)

        x = self.stage3(x)
        x = self.film3(x, film_feat)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        logits = self.head_class(x)
        aux_pred = self.head_aux(x)

        return logits, aux_pred

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


# -----------------------------------------------------------------------------
# 2. CactusResNet_MTL
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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


class CactusResNet_MTL(nn.Module):
    def __init__(self, num_classes=1, widths=[64, 128, 256]):
        super().__init__()

        # Stem: Stride 1 (Conservative)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.film0 = FiLMLayer(widths[0])

        # Stage 1
        self.stage1 = nn.Sequential(
            ResBlock(widths[0], widths[0]), ResBlock(widths[0], widths[0])
        )
        self.film1 = FiLMLayer(widths[0])

        # Stage 2
        self.stage2 = nn.Sequential(
            ResBlock(widths[0], widths[1], stride=2), ResBlock(widths[1], widths[1])
        )
        self.film2 = FiLMLayer(widths[1])

        # Stage 3
        self.stage3 = nn.Sequential(
            ResBlock(widths[1], widths[2], stride=2), ResBlock(widths[2], widths[2])
        )
        self.film3 = FiLMLayer(widths[2])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_class = nn.Linear(widths[2], num_classes)
        self.head_aux = nn.Linear(widths[2], 1)

    def forward(self, x, film_feat):
        x = self.stem(x)
        x = self.film0(x, film_feat)

        x = self.stage1(x)
        x = self.film1(x, film_feat)

        x = self.stage2(x)
        x = self.film2(x, film_feat)

        x = self.stage3(x)
        x = self.film3(x, film_feat)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        logits = self.head_class(x)
        aux_pred = self.head_aux(x)

        return logits, aux_pred


# -----------------------------------------------------------------------------
# 3. CactusNeXt_MTL
# -----------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # x: N, C, H, W -> N, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # -> N, C, H, W
        x = x.permute(0, 3, 1, 2)
        return x


class NeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise Conv 3x3
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm2d(dim)
        # Inverted Bottleneck: 1x1 -> 4x -> 1x1
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class CactusNeXt_MTL(nn.Module):
    def __init__(self, num_classes=1, widths=[64, 128, 256]):
        super().__init__()

        # Stem: Conv 3x3, stride 1 (Conservative)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(widths[0]),
        )
        self.film0 = FiLMLayer(widths[0])

        # Stage 1
        self.stage1 = nn.Sequential(NeXtBlock(widths[0]), NeXtBlock(widths[0]))
        self.film1 = FiLMLayer(widths[0])

        # Downsample 1: 32 -> 16
        self.down1 = nn.Sequential(
            LayerNorm2d(widths[0]),
            nn.Conv2d(widths[0], widths[1], kernel_size=2, stride=2),
        )

        # Stage 2
        self.stage2 = nn.Sequential(NeXtBlock(widths[1]), NeXtBlock(widths[1]))
        self.film2 = FiLMLayer(widths[1])

        # Downsample 2: 16 -> 8
        self.down2 = nn.Sequential(
            LayerNorm2d(widths[1]),
            nn.Conv2d(widths[1], widths[2], kernel_size=2, stride=2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(NeXtBlock(widths[2]), NeXtBlock(widths[2]))
        self.film3 = FiLMLayer(widths[2])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.norm_final = LayerNorm2d(widths[2])

        self.head_class = nn.Linear(widths[2], num_classes)
        self.head_aux = nn.Linear(widths[2], 1)

    def forward(self, x, film_feat):
        x = self.stem(x)
        x = self.film0(x, film_feat)

        x = self.stage1(x)
        x = self.film1(x, film_feat)

        x = self.down1(x)
        x = self.stage2(x)
        x = self.film2(x, film_feat)

        x = self.down2(x)
        x = self.stage3(x)
        x = self.film3(x, film_feat)

        x = self.norm_final(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        logits = self.head_class(x)
        aux_pred = self.head_aux(x)

        return logits, aux_pred
