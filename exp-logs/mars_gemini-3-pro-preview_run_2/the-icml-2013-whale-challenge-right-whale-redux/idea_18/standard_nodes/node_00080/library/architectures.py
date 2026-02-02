import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    p is a learnable parameter.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min to avoid NaN gradients for negative inputs (though usually inputs are ReLU'd)
        # or zeros when p < 1.
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

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


class WhaleClassifier(nn.Module):
    """
    Whale Call Detection Model.
    Adapts standard backbones (EfficientNet, ResNet) for 1-channel spectrogram inputs
    and uses GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleClassifier, self).__init__()

        # Load backbone without classification head and global pooling
        # This returns features of shape (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify the first layer to accept 1 channel instead of 3
        self._modify_first_layer(model_name)

        # Determine the number of output features from the backbone
        # We run a dummy forward pass to be architecture-agnostic
        with torch.no_grad():
            # Dummy input: (Batch=1, Channel=1, Freq=128, Time=64)
            dummy_input = torch.randn(1, 1, Config.N_MELS, Config.HOP_LENGTH)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Pooling Layer
        if Config.POOLING_TYPE == "gem":
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.fc = nn.Linear(in_features, 1)

    def _modify_first_layer(self, model_name):
        """
        Replaces the first convolutional layer with a 1-channel version.
        Weights are initialized by averaging the original 3-channel weights.
        """
        first_conv_name = None
        first_conv = None

        # Identify the first layer based on common naming conventions
        if "efficientnet" in model_name:
            first_conv_name = "conv_stem"
            if hasattr(self.backbone, "conv_stem"):
                first_conv = self.backbone.conv_stem
        elif "resnet" in model_name:
            first_conv_name = "conv1"
            if hasattr(self.backbone, "conv1"):
                first_conv = self.backbone.conv1

        # Fallback search if specific attribute not found
        if first_conv is None:
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_conv_name = name
                    first_conv = module
                    break

        if first_conv is None:
            raise ValueError(
                f"Could not find first convolutional layer for {model_name}"
            )

        # Create new Conv2d layer
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Initialize weights
        with torch.no_grad():
            if first_conv.weight.shape[1] == 3:
                # Average weights across the 3 input channels
                new_conv.weight[:] = torch.mean(first_conv.weight, dim=1, keepdim=True)
            else:
                # If for some reason it's already 1-channel or other, copy as is (broadcasting or direct)
                # This handles cases where we might reload a modified state dict
                if first_conv.weight.shape[1] == 1:
                    new_conv.weight[:] = first_conv.weight
                else:
                    # Fallback: take the first channel if dimensions mismatch differently
                    new_conv.weight[:] = first_conv.weight[:, 0:1, :, :]

            if first_conv.bias is not None:
                new_conv.bias[:] = first_conv.bias

        # Replace the layer in the backbone
        setattr(self.backbone, first_conv_name, new_conv)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)
        x = self.backbone(x)  # -> (Batch, C, H, W)
        x = self.pooling(x)  # -> (Batch, C, 1, 1)
        x = x.flatten(1)  # -> (Batch, C)
        x = self.fc(x)  # -> (Batch, 1)
        return x
