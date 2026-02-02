import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BlurPool(nn.Module):
    """
    Anti-Aliased Downsampling Layer (BlurPool).

    This layer applies a low-pass filter (blur) before subsampling to preserve
    shift-invariance and reduce aliasing, as described in:
    "Making Convolutional Networks Shift-Invariant Again" (Zhang et al., 2019).

    It is designed to replace standard strided operations (Max-Pooling or Strided Conv).
    """

    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, padding=0):
        """
        Args:
            channels (int): Number of input channels.
            pad_type (str): Padding type ('reflect', 'replicate', 'zero').
            filt_size (int): Size of the low-pass filter kernel (default: 3).
            stride (int): Downsampling stride (default: 2).
            padding (int): Explicit padding amount (default: 0).
        """
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_type = pad_type
        self.stride = stride
        self.channels = channels
        self.padding = padding

        # Create the smoothing kernel (Antialiasing filter)
        # Standard [1, 2, 1] kernel for size 3
        if self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            # General Pascal's triangle construction could go here,
            # but 3 is standard for ResNet replacement.
            raise ValueError(f"Unsupported filter size: {self.filt_size}")

        # Create 2D filter: outer product
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)  # Normalize

        # Reshape for depthwise convolution: (Out, In/Groups, H, W)
        # Groups = Channels, so shape is (Channels, 1, H, W)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, x):
        """
        Applies blur and downsampling.
        """
        # Calculate padding to preserve size if stride were 1
        # For k=3, pad=1. For k=5, pad=2.
        if self.padding == 0:
            pad_size = self.filt_size // 2
        else:
            pad_size = self.padding

        if self.pad_type == "reflect":
            x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        elif self.pad_type == "replicate":
            x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="replicate")
        else:
            # Zero padding is handled implicitly by conv2d if we passed padding arg,
            # but to be consistent with other modes we pad manually here or use conv2d padding.
            # Here we use conv2d padding for zero pad if pad_type is not special
            x_pad = x

        if self.pad_type in ["reflect", "replicate"]:
            # Padding already done
            return F.conv2d(x_pad, self.filt, stride=self.stride, groups=self.channels)
        else:
            # Apply zero padding via conv2d
            return F.conv2d(
                x, self.filt, stride=self.stride, padding=pad_size, groups=self.channels
            )
