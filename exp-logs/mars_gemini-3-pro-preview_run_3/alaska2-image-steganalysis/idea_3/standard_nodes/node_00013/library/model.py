import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


class HPF(nn.Module):
    """
    High-Pass Filter Layer.
    Initialized with the fixed 5x5 KV kernel to extract noise residuals.
    This layer is non-trainable.
    """

    def __init__(self):
        super(HPF, self).__init__()
        # KV Kernel definition
        # Shape: (5, 5)
        kv_kernel = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )

        # Reshape to (Out_channels, In_channels, H, W) -> (1, 1, 5, 5)
        kv_kernel = torch.from_numpy(kv_kernel).unsqueeze(0).unsqueeze(0)

        self.conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.conv.weight.data = kv_kernel

        # Freeze parameters
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (Batch, 1, H, W)
        return self.conv(x)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes the generalized mean of the feature map activations.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (Batch, Channels, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply clamping to avoid NaN gradients for negative inputs (though typically ReLU precedes this)
        # or zero inputs when p < 1.
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class MonoResidualEfficientNet(nn.Module):
    """
    Mono-Residual EfficientNet-B2 with GeM Pooling.

    Architecture:
    1. Input (Y channel) -> HPF Layer (Residuals)
    2. Residuals -> EfficientNet-B2 Backbone (Modified 1-channel input)
    3. Features -> GeM Pooling -> Linear Head
    """

    def __init__(self, model_name="efficientnet_b2", pretrained=True, num_classes=1):
        super(MonoResidualEfficientNet, self).__init__()

        # 1. HPF Preprocessing
        self.hpf = HPF()

        # 2. Backbone
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # 3. Modify First Convolutional Layer
        # The standard first layer in EfficientNet is named 'conv_stem'
        if hasattr(self.backbone, "conv_stem"):
            original_conv = self.backbone.conv_stem

            # Create a new Conv2d layer with 1 input channel instead of 3
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )

            # Initialize weights: Sum the weights across the RGB channel dimension
            # Original weight shape: (Out, 3, K, K)
            # New weight shape: (Out, 1, K, K)
            with torch.no_grad():
                new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))
                if original_conv.bias is not None:
                    new_conv.bias.copy_(original_conv.bias)

            # Replace the layer in the backbone
            self.backbone.conv_stem = new_conv
        else:
            raise AttributeError(
                f"The provided model {model_name} does not have a 'conv_stem' layer."
            )

        # 4. GeM Pooling
        self.gem = GeM()

        # 5. Classification Head
        # Get the number of features from the backbone
        self.in_features = self.backbone.num_features
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.
        x: Input tensor of shape (Batch, 1, Height, Width)
        """
        # 1. Extract Noise Residuals
        x = self.hpf(x)

        # 2. Backbone Feature Extraction
        # forward_features returns the feature maps before global pooling
        # Shape: (Batch, Channels, H', W')
        x = self.backbone.forward_features(x)

        # 3. GeM Pooling
        # Shape: (Batch, Channels, 1, 1)
        x = self.gem(x)

        # 4. Flatten
        # Shape: (Batch, Channels)
        x = x.flatten(1)

        # 5. Classification
        # Shape: (Batch, num_classes)
        x = self.fc(x)

        return x
