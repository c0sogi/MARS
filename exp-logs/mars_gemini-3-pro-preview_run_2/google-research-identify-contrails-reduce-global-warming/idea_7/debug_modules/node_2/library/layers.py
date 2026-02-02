import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 Layer.

    Wraps torchvision.ops.deform_conv2d with internal offset and mask prediction layers.
    This allows the network to learn to deform the sampling grid of the convolution,
    adapting to the geometric variations of targets (e.g., linear contrails).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(DeformableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Standard convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Offset and Mask predictor
        # For each element in the kernel, we need 2 offsets (x, y) and 1 mask
        # Total channels = groups * kernel_size * kernel_size * 3
        # However, usually offsets are shared across input channels (or per group).
        # Standard implementation: 1 offset group implies offsets are shared.
        # We assume offset_groups = 1 for simplicity and stability in this architecture.

        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.reset_parameters()
        self._init_offset_mask_conv()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def _init_offset_mask_conv(self):
        # Initialize offsets to 0 and masks to result in ~0.5 (sigmoid(0))
        # This ensures the layer starts training behaving like a standard convolution
        nn.init.constant_(self.offset_mask_conv.weight, 0.0)
        nn.init.constant_(self.offset_mask_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict offsets and masks
        out = self.offset_mask_conv(x)

        # Split output into offsets (2 coords) and mask (1 weight)
        # Shape: [B, 3 * K*K, H, W]
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # Concatenate x and y offsets
        offset = torch.cat((o1, o2), dim=1)

        # Apply sigmoid to mask to get range [0, 1]
        mask = torch.sigmoid(mask)

        # Apply Deformable Convolution
        output = ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

        return output


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Module.

    Concurrent Spatial and Channel 'Squeeze & Excitation' in Fully Convolutional Networks.
    Enhances meaningful features by recalibrating feature maps spatially and channel-wise.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(SCSEModule, self).__init__()

        # Channel Squeeze and Excitation (cSE)
        # Global Avg Pool -> Dense -> ReLU -> Dense -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        # Conv 1x1 -> Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        # Channel attention: recalibrate channels
        # cSE output: [B, C, 1, 1] broadcasted to [B, C, H, W]
        x_chn = x * self.cSE(x)

        # Spatial attention: recalibrate spatial positions
        # sSE output: [B, 1, H, W] broadcasted to [B, C, H, W]
        x_spa = x * self.sSE(x)

        # Combine (Max-out or Add, usually Add is preferred in SCSE paper)
        return x_chn + x_spa


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).

    Captures multi-scale information by probing the incoming feature map with filters
    at multiple sampling rates and effective fields-of-views.
    """

    def __init__(
        self, in_channels: int, out_channels: int, atrous_rates: list = [6, 12, 18]
    ):
        super(ASPP, self).__init__()

        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous Convolutions
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Image Pooling Branch (Global Context)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(modules)

        # Final projection
        # Input channels = (len(atrous_rates) + 1 (1x1 conv) + 1 (pooling)) * out_channels
        n_branches = len(atrous_rates) + 2
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []

        # Apply convolutional branches
        for conv in self.convs:
            res.append(conv(x))

        # Apply image pooling branch
        # Upsample back to input size
        pool = self.image_pooling(x)
        pool = F.interpolate(
            pool, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(pool)

        # Concatenate all branches
        res = torch.cat(res, dim=1)

        # Final projection
        return self.project(res)
