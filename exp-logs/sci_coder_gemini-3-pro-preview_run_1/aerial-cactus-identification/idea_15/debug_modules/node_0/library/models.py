import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# -------------------------------------------------------------------------
# Components
# -------------------------------------------------------------------------


class FiLMGenerator(nn.Module):
    """
    Generates scale (gamma) and shift (beta) parameters from file size metadata.
    """

    def __init__(self, input_dim=1, channels=64):
        super(FiLMGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2 * channels)
        )
        self.channels = channels

        # Initialize to identity modulation (gamma=1, beta=0)
        with torch.no_grad():
            self.net[2].weight.data.fill_(0)
            self.net[2].bias.data.fill_(0)
            # Set bias for gamma to 1
            self.net[2].bias.data[:channels] = 1.0

    def forward(self, fsize):
        # fsize: (B,) or (B, 1)
        if fsize.dim() == 1:
            fsize = fsize.unsqueeze(1)

        # out: (B, 2*C)
        out = self.net(fsize)
        gamma = out[:, : self.channels].unsqueeze(2).unsqueeze(3)
        beta = out[:, self.channels :].unsqueeze(2).unsqueeze(3)
        return gamma, beta


class ConservativeStem(nn.Module):
    """
    3x3 Conv, Stride 1, No Pooling. Preserves 32x32 resolution.
    """

    def __init__(self, in_channels, out_channels):
        super(ConservativeStem, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AuxHead(nn.Module):
    """
    Auxiliary Classification Head with FiLM modulation.
    """

    def __init__(self, in_channels, num_classes=1):
        super(AuxHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.film_gen = FiLMGenerator(channels=in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x, fsize):
        x = self.pool(x)  # (B, C, 1, 1)
        gamma, beta = self.film_gen(fsize)
        x = x * gamma + beta
        x = x.flatten(1)
        return self.fc(x)


# -------------------------------------------------------------------------
# Blocks
# -------------------------------------------------------------------------


class RepVGGBlock(nn.Module):
    """
    RepVGG Block: Multi-branch training, fused inference.
    Includes FiLM modulation.
    """

    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU(inplace=True)
        self.film_gen = FiLMGenerator(channels=out_channels)

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

    def forward(self, x, fsize):
        if self.deploy:
            out = self.rbr_reparam(x)
        else:
            out = self.rbr_dense(x) + self.rbr_1x1(x)
            if self.rbr_identity is not None:
                out += self.rbr_identity(x)

        # Apply FiLM before activation
        gamma, beta = self.film_gen(fsize)
        out = out * gamma + beta
        return self.act(out)

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
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

    def _get_equivalent_kernel_bias(self):
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
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # Identity branch is just BN
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


class ResNetBlock(nn.Module):
    """
    Standard ResNet BasicBlock with FiLM modulation.
    """

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

        self.film_gen = FiLMGenerator(channels=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, fsize):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply FiLM to the residual branch before addition
        gamma, beta = self.film_gen(fsize)
        out = out * gamma + beta

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class LayerNorm(nn.Module):
    """Channel-first LayerNorm for ConvNeXt style blocks"""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NeXtBlock(nn.Module):
    """
    ConvNeXt-style block but with strictly 3x3 kernels.
    Includes FiLM modulation.
    """

    def __init__(self, dim, drop_path=0.0):
        super(NeXtBlock, self).__init__()
        # Depthwise 3x3
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)

        # FiLM Generator
        self.film_gen = FiLMGenerator(channels=dim)

        # Pointwise expansions
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = nn.Identity()  # Placeholder for simplicity

    def forward(self, x, fsize):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Apply FiLM after normalization
        gamma, beta = self.film_gen(fsize)
        x = x * gamma + beta

        # Pointwise ops (implemented with Linear, requiring permute)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x = input + self.drop_path(x)
        return x


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------


class RepVGG_FiLM(nn.Module):
    def __init__(self, num_classes=1, deploy=False):
        super(RepVGG_FiLM, self).__init__()
        self.deploy = deploy

        # Channels schedule
        self.stages = [64, 128, 256, 512]

        self.stem = ConservativeStem(3, self.stages[0])

        # Stage 1: 32x32
        self.layer1 = self._make_layer(self.stages[0], self.stages[0], 2, stride=1)
        # Stage 2: 16x16
        self.layer2 = self._make_layer(self.stages[0], self.stages[1], 3, stride=2)
        # Stage 3: 8x8
        self.layer3 = self._make_layer(self.stages[1], self.stages[2], 3, stride=2)
        # Stage 4: 4x4
        self.layer4 = self._make_layer(self.stages[2], self.stages[3], 3, stride=2)

        self.aux_head = AuxHead(self.stages[2], num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(
            RepVGGBlock(in_channels, out_channels, stride, deploy=self.deploy)
        )
        for _ in range(1, blocks):
            layers.append(
                RepVGGBlock(out_channels, out_channels, 1, deploy=self.deploy)
            )
        return nn.ModuleList(layers)

    def forward(self, x, fsize):
        x = self.stem(x)

        for block in self.layer1:
            x = block(x, fsize)
        for block in self.layer2:
            x = block(x, fsize)

        # Stage 3
        for block in self.layer3:
            x = block(x, fsize)
        aux = self.aux_head(x, fsize)

        # Stage 4
        for block in self.layer4:
            x = block(x, fsize)

        x = self.pool(x).flatten(1)
        out = self.fc(x)

        return {"logits": out, "aux_logits": aux}


class ResNet_FiLM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet_FiLM, self).__init__()

        self.stages = [64, 128, 256, 512]
        self.stem = ConservativeStem(3, self.stages[0])

        self.layer1 = self._make_layer(self.stages[0], self.stages[0], 2, stride=1)
        self.layer2 = self._make_layer(self.stages[0], self.stages[1], 2, stride=2)
        self.layer3 = self._make_layer(self.stages[1], self.stages[2], 2, stride=2)
        self.layer4 = self._make_layer(self.stages[2], self.stages[3], 2, stride=2)

        self.aux_head = AuxHead(self.stages[2], num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.ModuleList(layers)

    def forward(self, x, fsize):
        x = self.stem(x)

        for block in self.layer1:
            x = block(x, fsize)
        for block in self.layer2:
            x = block(x, fsize)

        for block in self.layer3:
            x = block(x, fsize)
        aux = self.aux_head(x, fsize)

        for block in self.layer4:
            x = block(x, fsize)

        x = self.pool(x).flatten(1)
        out = self.fc(x)

        return {"logits": out, "aux_logits": aux}


class NeXt_FiLM(nn.Module):
    def __init__(self, num_classes=1):
        super(NeXt_FiLM, self).__init__()

        self.stages = [64, 128, 256, 512]
        self.stem = ConservativeStem(3, self.stages[0])

        # Downsampling layers between stages (except stage 1)
        self.downsample2 = nn.Sequential(
            LayerNorm(self.stages[0], eps=1e-6),
            nn.Conv2d(self.stages[0], self.stages[1], kernel_size=2, stride=2),
        )
        self.downsample3 = nn.Sequential(
            LayerNorm(self.stages[1], eps=1e-6),
            nn.Conv2d(self.stages[1], self.stages[2], kernel_size=2, stride=2),
        )
        self.downsample4 = nn.Sequential(
            LayerNorm(self.stages[2], eps=1e-6),
            nn.Conv2d(self.stages[2], self.stages[3], kernel_size=2, stride=2),
        )

        self.layer1 = self._make_layer(self.stages[0], 2)
        self.layer2 = self._make_layer(self.stages[1], 2)
        self.layer3 = self._make_layer(self.stages[2], 2)
        self.layer4 = self._make_layer(self.stages[3], 2)

        self.aux_head = AuxHead(self.stages[2], num_classes)
        self.norm = LayerNorm(self.stages[3], eps=1e-6)
        self.fc = nn.Linear(self.stages[3], num_classes)

    def _make_layer(self, dim, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(NeXtBlock(dim))
        return nn.ModuleList(layers)

    def forward(self, x, fsize):
        x = self.stem(x)

        # Stage 1
        for block in self.layer1:
            x = block(x, fsize)

        # Stage 2
        x = self.downsample2(x)
        for block in self.layer2:
            x = block(x, fsize)

        # Stage 3
        x = self.downsample3(x)
        for block in self.layer3:
            x = block(x, fsize)
        aux = self.aux_head(x, fsize)

        # Stage 4
        x = self.downsample4(x)
        for block in self.layer4:
            x = block(x, fsize)

        x = self.norm(x)
        x = x.mean([-2, -1])  # Global Avg Pool
        out = self.fc(x)

        return {"logits": out, "aux_logits": aux}


def reparameterize_model(model):
    """
    Recursively switches all RepVGG blocks in the model to deploy mode.
    """
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    return model
