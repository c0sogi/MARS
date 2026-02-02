import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ARCH, PRETRAINED, NUM_CLASSES, IN_CHANNELS


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp inputs to eps to avoid NaN gradients with fractional powers
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


class WhaleModel(nn.Module):
    def __init__(
        self,
        arch=ARCH,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
    ):
        super(WhaleModel, self).__init__()

        # Load the backbone with no classifier and no global pooling
        # This returns the feature maps directly
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify the first convolutional layer to accept 1-channel input
        # For EfficientNet, the first layer is named 'conv_stem'
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Initialize new weights by averaging the original RGB weights
            # Shape: (out_channels, in_channels, k, k)
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)

            self.backbone.conv_stem = new_conv

        elif hasattr(self.backbone, "conv1"):
            # Fallback for ResNet-like architectures
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            self.backbone.conv1 = new_conv

        # Generalized Mean Pooling
        self.pooling = GeM()

        # Determine the number of input features for the linear head
        # We perform a dummy forward pass to dynamically determine the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Classification Head
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Extract features: (Batch, Channels, Height, Width)
        x = self.backbone(x)

        # Apply GeM Pooling: (Batch, Channels, 1, 1)
        x = self.pooling(x)

        # Flatten: (Batch, Channels)
        x = x.flatten(1)

        # Classification: (Batch, Num_Classes)
        x = self.head(x)
        return x


def get_model(
    arch=ARCH, pretrained=PRETRAINED, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS
):
    """
    Factory function to create the model.

    Args:
        arch (str): Name of the timm architecture.
        pretrained (bool): Whether to load pretrained ImageNet weights.
        num_classes (int): Number of output classes.
        in_channels (int): Number of input audio channels.

    Returns:
        nn.Module: The configured WhaleModel.
    """
    model = WhaleModel(
        arch=arch,
        pretrained=pretrained,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    return model
