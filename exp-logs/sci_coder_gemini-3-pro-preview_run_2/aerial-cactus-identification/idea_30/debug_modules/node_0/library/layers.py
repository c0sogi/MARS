import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    References:
        Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks", CVPR 2020.
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # Adaptive kernel size calculation
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)

        # Reshape for 1D Conv: (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)  # (B, 1, C)

        # Reshape back: (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)
        return x * y.expand_as(x)


class BlurPool(nn.Module):
    """
    BlurPool layer for anti-aliased downsampling.
    Applies a low-pass filter (smoothing) before subsampling.

    References:
        Zhang, Richard. "Making convolutional networks shift-invariant again." ICML 2019.
    """

    def __init__(self, channels, stride=2, filt_size=3):
        super(BlurPool, self).__init__()
        self.channels = channels
        self.stride = stride

        # Create the smoothing kernel (binomial filter)
        if filt_size == 3:
            filt = torch.tensor([1.0, 2.0, 1.0])
        elif filt_size == 5:
            filt = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError("BlurPool filter size must be 3 or 5")

        filt = filt / torch.sum(filt)
        # Create 2D kernel by outer product
        filt_2d = filt[:, None] * filt[None, :]
        # Expand dims to match Conv2d weight shape: (Out, In/Groups, kH, kW)
        # Groups = channels, so In/Groups = 1
        filt_2d = filt_2d[None, None, :, :].repeat((self.channels, 1, 1, 1))

        self.register_buffer("filt", filt_2d)
        self.pad_size = filt_size // 2

    def forward(self, x):
        # Pad input to maintain size before striding if necessary,
        # or simply pad to handle boundary conditions for the filter.
        # For stride=2 downsampling, we typically want to preserve information
        # at the edges similar to 'same' padding logic before the stride.
        # Reflection padding is often used to reduce boundary artifacts.
        x_pad = F.pad(
            x,
            (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode="reflect",
        )

        return F.conv2d(x_pad, self.filt, stride=self.stride, groups=self.channels)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).

    Computes f = (mean(x^p))^(1/p).
    p is a learnable parameter.

    References:
        Radenović et al., "Fine-tuning CNN Image Retrieval with No Human Annotation", TPAMI 2018.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Clamp to avoid NaN gradients or undefined operations with negative values (though ReLU usually precedes)
        x = x.clamp(min=self.eps)

        # Average pooling on x^p
        # (B, C, H, W) -> (B, C, 1, 1)
        x_pow = x.pow(self.p)
        avg_x_pow = F.avg_pool2d(x_pow, (x_pow.size(-2), x_pow.size(-1)))

        # Raise to power 1/p
        gem = avg_x_pow.pow(1.0 / self.p)

        # Flatten to (B, C)
        return gem.flatten(1)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
