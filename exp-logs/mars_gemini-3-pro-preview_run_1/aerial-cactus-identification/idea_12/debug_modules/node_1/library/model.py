import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper to create a Conv2d followed by BatchNorm2d.
    """
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
    """
    RepVGG Block that supports structural re-parameterization.
    Training: 3x3 Branch + 1x1 Branch + Identity Branch
    Inference: Single 3x3 Conv
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
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
        self.groups = groups

        # Activation
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
            # 3x3 Branch
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            # 1x1 Branch
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding - kernel_size // 2,
                groups=groups,
            )
            # Identity Branch (only if dimensions match)
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Derives the fused 3x3 kernel and bias from the multi-branch structure.
        """
        # 1. Fuse 3x3 Branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 2. Fuse 1x1 Branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # Pad 1x1 kernel to 3x3
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)

        # 3. Fuse Identity Branch
        kernelid, biasid = self._get_kernel_bias_from_id_tensor()

        # Sum params
        return (
            kernel3x3 + kernel1x1 + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _get_kernel_bias_from_id_tensor(self):
        if self.rbr_identity is None:
            return 0, 0

        # Identity is just a BN. We need to convert it to a Conv.
        # The equivalent conv kernel for identity is 1s on the diagonal of the center spatial pos.
        kernel_value = np.zeros(
            (self.in_channels, self.in_channels, 3, 3), dtype=np.float32
        )
        for i in range(self.in_channels):
            kernel_value[i, i, 1, 1] = 1

        # Convert to tensor
        id_tensor = torch.from_numpy(kernel_value).to(self.rbr_identity.weight.device)

        # Fuse the BN into this identity conv
        kernel, bias = self._fuse_bn_tensor(self.rbr_identity, id_tensor)
        return kernel, bias

    def _fuse_bn_tensor(self, branch, input_kernel=None):
        """
        Fuses BatchNorm into Convolution or generates Conv from BN.
        """
        if isinstance(branch, nn.Sequential):
            # It's a Conv + BN
            conv = branch.conv
            bn = branch.bn
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            # It's just a BN (Identity branch)
            assert isinstance(branch, nn.BatchNorm2d)
            if input_kernel is None:
                # Should have been provided by _get_kernel_bias_from_id_tensor
                return 0, 0
            kernel = input_kernel
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # Fused kernel: W_conv * (gamma / sigma)
        fused_kernel = kernel * t

        # Fused bias: beta - mu * (gamma / sigma)
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias

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

        # Remove branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class GatingHead(nn.Module):
    """
    MLP that computes a gating vector from metadata (file size).
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatingHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output range [0, 1] for gating
        )

    def forward(self, metadata):
        return self.mlp(metadata)


class MetadataGatedRepVGG(nn.Module):
    """
    RepVGG Backbone with Metadata-based Channel Gating.
    """

    def __init__(self):
        super(MetadataGatedRepVGG, self).__init__()

        # --- Conservative Stem ---
        # 3x3 Conv, Stride 1, No Pooling. Preserves 32x32 resolution.
        self.stem = RepVGGBlock(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.BASE_WIDTH,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=False,
        )

        # --- Backbone Stages ---
        self.stages = nn.ModuleList()
        in_channels = Config.BASE_WIDTH

        # Build stages based on Config
        # Resolution flow: 32x32 -> 16x16 -> 8x8 -> 4x4
        for depth, width in zip(Config.STAGE_DEPTHS, Config.STAGE_WIDTHS):
            stage_blocks = []

            # First block of the stage handles downsampling (Stride 2)
            # Except if we wanted to keep resolution, but standard RepVGG downsamples at start of stage.
            # Given 32x32 input and 3 stages, we want 3 downsamples (16, 8, 4).
            # So stride=2 for the first block of each stage is correct.
            stage_blocks.append(
                RepVGGBlock(
                    in_channels=in_channels, out_channels=width, stride=2, deploy=False
                )
            )

            # Remaining blocks in the stage (Stride 1)
            for _ in range(depth - 1):
                stage_blocks.append(
                    RepVGGBlock(
                        in_channels=width, out_channels=width, stride=1, deploy=False
                    )
                )

            self.stages.append(nn.Sequential(*stage_blocks))
            in_channels = width

        self.final_channels = in_channels

        # --- Global Pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Metadata Gating Head ---
        self.gate = GatingHead(
            input_dim=Config.METADATA_DIM,
            hidden_dim=Config.GATE_HIDDEN_DIM,
            output_dim=self.final_channels,
        )

        # --- Classifier ---
        self.linear = nn.Linear(self.final_channels, Config.NUM_CLASSES)

    def forward(self, x, metadata):
        # 1. Visual Feature Extraction
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # 2. Pooling (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # 3. Metadata Gating
        # metadata shape: (B, 1)
        # gate_vector shape: (B, C)
        gate_vector = self.gate(metadata)

        # Element-wise multiplication (recalibration)
        x = x * gate_vector

        # 4. Classification
        x = self.linear(x)
        return x

    def switch_to_deploy(self):
        """
        Recursively switches all RepVGG blocks to inference mode.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
