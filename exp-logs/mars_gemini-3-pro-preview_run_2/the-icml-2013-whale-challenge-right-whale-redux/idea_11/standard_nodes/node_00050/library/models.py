import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the p-th power of the input, applies average pooling, and then takes the (1/p)-th power.
    Useful for capturing salient features in the presence of background noise.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp input to avoid NaN when taking power
        x = torch.clamp(x, min=eps)
        # Apply Average Pooling on x^p over the spatial dimensions
        # Output shape: (B, C, 1, 1)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1)))
        # Take the inverse power
        return x.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Wrapper for timm backbones (EfficientNet-B0, ResNet34) adapted for 1-channel audio spectrograms.
    Implements the 'Physically-Aligned' strategy by averaging RGB weights for the first layer
    and using GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()
        self.model_name = model_name

        # Load the backbone from timm
        # global_pool='' ensures we get the spatial feature maps (B, C, H, W)
        # num_classes=0 removes the original classification head
        # in_chans=3 loads the standard RGB pretrained weights, which we will manually adapt
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )

        # Modify the first convolutional layer to accept 1-channel input
        self._modify_first_layer()

        # Determine the number of output channels from the backbone
        self.in_features = self.backbone.num_features

        # Replace global pooling with GeM
        self.global_pool = GeM(p=3.0)

        # New classification head
        self.head = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (3 channels) with a 1-channel layer.
        Weights are initialized by averaging the original RGB weights to preserve spatial filters.
        """
        module_to_modify = None
        name_to_modify = None

        # Identify the first conv layer based on common naming conventions
        if hasattr(self.backbone, "conv1"):
            # Standard for ResNet
            module_to_modify = self.backbone.conv1
            name_to_modify = "conv1"
        elif hasattr(self.backbone, "conv_stem"):
            # Standard for EfficientNet
            module_to_modify = self.backbone.conv_stem
            name_to_modify = "conv_stem"
        else:
            # Fallback: search for the first Conv2d module
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    module_to_modify = module
                    name_to_modify = name
                    break

        if module_to_modify is None:
            raise ValueError(f"Could not find first Conv2d layer in {self.model_name}")

        # Create a new Conv2d layer with in_channels=1
        old_conv = module_to_modify
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
        )

        # Initialize weights: Average the 3 RGB channels into 1
        # old_conv.weight shape: (Out, 3, H, W) -> new_conv.weight shape: (Out, 1, H, W)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        # Replace the layer in the backbone
        if hasattr(self.backbone, name_to_modify):
            setattr(self.backbone, name_to_modify, new_conv)
        else:
            # Handle cases where the layer might be nested (less common for stem)
            # For this task's specific models (ResNet/EffNet), setattr on backbone is sufficient
            pass

    def forward(self, x):
        # x shape: (B, 1, H, W)

        # Extract features
        x = self.backbone(x)
        # x shape: (B, C, H', W')

        # Apply GeM Pooling
        x = self.global_pool(x)
        # x shape: (B, C, 1, 1)

        # Flatten
        x = x.flatten(1)
        # x shape: (B, C)

        # Classification
        logits = self.head(x)
        # logits shape: (B, 1)

        return logits


def get_model(model_name, pretrained=True):
    """
    Factory function to create a WhaleModel instance.

    Args:
        model_name (str): Name of the architecture (e.g., 'efficientnet_b0', 'resnet34').
        pretrained (bool): Whether to load ImageNet weights.

    Returns:
        WhaleModel: The initialized model.
    """
    return WhaleModel(model_name, pretrained=pretrained)
