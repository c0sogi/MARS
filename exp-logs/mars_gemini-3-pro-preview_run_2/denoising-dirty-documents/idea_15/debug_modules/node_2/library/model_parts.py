import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Decomposes attention into two 1D feature encoding processes (horizontal and vertical)
    to preserve positional information.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # Using SiLU as per strategy

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool along H and W axes
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale context using dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[0],
                dilation=rates[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[1],
                dilation=rates[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[2],
                dilation=rates[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]

        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)

        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode="bilinear", align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)
        return out


class RepBlock(nn.Module):
    """
    Reparameterized Block (RepBlock).

    Training Topology:
        - 3x3 Conv + BN
        - 1x1 Conv + BN
        - Identity + BN (if dimensions match)

    Inference Topology:
        - Fused 3x3 Conv

    This allows for high capacity and gradient flow during training,
    but collapses into a single efficient convolution during inference.
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
        deploy=False,
    ):
        super(RepBlock, self).__init__()

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Activation
        self.non_linearity = nn.SiLU()

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
            # 1. 3x3 Branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # 2. 1x1 Branch
            # Padding is 0 for 1x1 conv
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # 3. Identity Branch
            # Only if input/output dimensions and spatial resolution match
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

    def forward(self, inputs):
        if self.deploy:
            return self.non_linearity(self.rbr_reparam(inputs))

        # Training forward pass: Sum of branches
        id_out = 0
        if self.rbr_identity is not None:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    def get_equivalent_kernel_bias(self):
        """
        Calculates the equivalent 3x3 kernel and bias by fusing all branches.
        """
        # 1. Fuse 3x3 Branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 2. Fuse 1x1 Branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        # 3. Fuse Identity Branch
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # Add them up
        # We need to pad the 1x1 kernel to 3x3
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """
        Pads a 1x1 kernel to 3x3.
        """
        if kernel1x1 is None:
            return 0
        # Pad (left, right, top, bottom) -> (1, 1, 1, 1)
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuses BatchNorm into the preceding Convolution or creates a kernel for Identity BN.
        Returns (kernel, bias).
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            # It's a Conv + BN sequence
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # It's just a BatchNorm layer (Identity branch)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                # Create an identity kernel: 1 at the center, 0 elsewhere
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
        """
        Converts the multi-branch structure into a single 3x3 convolution.
        Should be called before inference.
        """
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            dilation=self.rbr_dense[0].dilation,
            groups=self.rbr_dense[0].groups,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches to save memory
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.deploy = True
