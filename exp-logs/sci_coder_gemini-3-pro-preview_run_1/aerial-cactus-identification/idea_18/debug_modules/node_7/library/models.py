import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


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


class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation Generator.
    Projects a scalar (normalized file size) into modulation parameters (gamma, beta)
    for multiple feature maps.
    """

    def __init__(self, input_dim=1, total_channels=0, hidden_dim=64):
        super(FiLMGenerator, self).__init__()
        self.total_channels = total_channels

        # Simple MLP: Input -> Linear -> ReLU -> Linear -> (Gamma, Beta)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_channels * 2),
        )

        # Initialize last layer to produce identity modulation (gamma=1, beta=0)
        # This ensures the model starts with standard behavior
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)
        # We will add 1 to gamma during forward pass

    def forward(self, x):
        # x: (B, 1) or (B,)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = self.net(x)  # (B, total_channels * 2)

        # Split into gamma and beta
        gamma, beta = torch.split(out, self.total_channels, dim=1)

        # Add 1 to gamma so initial state is identity (x * 1 + 0)
        gamma = gamma + 1.0

        return gamma, beta


class RepVGGBlock(nn.Module):
    """
    RepVGG Block: Multi-branch training, Single-branch inference.
    Branches: 3x3 Conv, 1x1 Conv, Identity.
    """

    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.nonlinearity = nn.ReLU()

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
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        # Fuse 3x3
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        # Fuse 1x1
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # Pad 1x1 to 3x3
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)
        # Fuse Identity
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

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
        else:  # Identity branch is just BN
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


class ResNetBlock(nn.Module):
    """
    Standard ResNet BasicBlock with 3x3 convolutions.
    """

    expansion = 1

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

    def switch_to_deploy(self):
        # No re-param for ResNet, but method needed for interface consistency
        pass


class CactusModel(nn.Module):
    def __init__(
        self, arch="RepVGG", in_chans=3, num_classes=1, use_film=True, use_mtl=True
    ):
        super(CactusModel, self).__init__()
        self.arch = arch
        self.use_film = use_film
        self.use_mtl = use_mtl

        # Architecture Config (conservative downsampling for 32x32)
        # Stem -> Stage1(32x32) -> Stage2(16x16) -> Stage3(8x8) -> Stage4(4x4)
        self.stage_planes = [64, 64, 128, 256, 512]
        self.strides = [1, 1, 2, 2, 2]  # Stride for Stem, S1, S2, S3, S4

        # 1. Build Stem
        if arch == "RepVGG":
            self.stem = RepVGGBlock(
                in_chans, self.stage_planes[0], stride=self.strides[0]
            )
        else:
            # ResNet Stem: 3x3 conv, stride 1 (no pooling)
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    self.stage_planes[0],
                    kernel_size=3,
                    stride=self.strides[0],
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.stage_planes[0]),
                nn.ReLU(inplace=True),
            )

        # 2. Build Stages
        self.stages = nn.ModuleList()
        in_planes = self.stage_planes[0]

        for i in range(1, len(self.stage_planes)):
            out_planes = self.stage_planes[i]
            stride = self.strides[i]

            # Number of blocks per stage
            num_blocks = 2

            layers = []
            # First block handles stride and channel change
            if arch == "RepVGG":
                layers.append(RepVGGBlock(in_planes, out_planes, stride=stride))
                for _ in range(1, num_blocks):
                    layers.append(RepVGGBlock(out_planes, out_planes, stride=1))
            else:  # ResNet
                layers.append(ResNetBlock(in_planes, out_planes, stride=stride))
                for _ in range(1, num_blocks):
                    layers.append(ResNetBlock(out_planes, out_planes, stride=1))

            self.stages.append(nn.Sequential(*layers))
            in_planes = out_planes

        # 3. FiLM Generator
        if self.use_film:
            # We modulate the output of Stem + 4 Stages
            total_channels = sum(self.stage_planes)
            self.film_generator = FiLMGenerator(total_channels=total_channels)

            # Pre-calculate indices for splitting the flat gamma/beta vectors
            self.split_indices = self.stage_planes

        # 4. Heads
        final_dim = self.stage_planes[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_dim, num_classes)

        if self.use_mtl:
            self.fc_mtl = nn.Linear(final_dim, 1)

    def forward(self, x, file_size_norm=None):
        """
        Args:
            x: Input images (B, C, H, W)
            file_size_norm: Normalized file sizes (B,) or (B, 1)
        """
        # Generate FiLM params if needed
        gammas, betas = [], []
        if self.use_film and file_size_norm is not None:
            g_flat, b_flat = self.film_generator(file_size_norm)
            gammas = torch.split(g_flat, self.split_indices, dim=1)
            betas = torch.split(b_flat, self.split_indices, dim=1)

        # Helper to apply FiLM
        def apply_film(feat, idx):
            if self.use_film and file_size_norm is not None:
                g = gammas[idx].unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
                b = betas[idx].unsqueeze(2).unsqueeze(3)
                return feat * g + b
            return feat

        # Forward Pass

        # Stem
        out = self.stem(x)
        out = apply_film(out, 0)

        # Stages
        for i, stage in enumerate(self.stages):
            out = stage(out)
            out = apply_film(out, i + 1)

        # Global Pooling
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        # Heads
        logits = self.fc(out)

        output = {"logits": logits}

        if self.use_mtl:
            mtl_pred = self.fc_mtl(out)
            output["mtl_pred"] = mtl_pred

        return output

    def switch_to_deploy(self):
        """
        Recursively calls switch_to_deploy on all submodules (mainly for RepVGG).
        """
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
