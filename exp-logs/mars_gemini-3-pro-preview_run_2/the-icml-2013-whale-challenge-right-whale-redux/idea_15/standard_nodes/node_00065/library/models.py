import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import GeM


class WhaleModel(nn.Module):
    """
    Level 0 Model for Right Whale Call Detection.

    This class wraps a timm backbone (EfficientNet or ResNet), adapts the first
    convolutional layer to accept 1-channel (grayscale) input by averaging the
    pretrained RGB weights, and replaces the global pooling/classifier with
    GeM pooling and a linear output layer.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture to load via timm.
            pretrained (bool): Whether to load pretrained ImageNet/JFT weights.
        """
        super(WhaleModel, self).__init__()

        # Load backbone with 3 channels initially to retrieve original RGB weights.
        # We set num_classes=0 and global_pool='' to get the raw spatial feature map.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )

        # Adapt the first layer to 1 channel
        self._adapt_first_layer()

        # Define Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Define Classification Head
        # self.backbone.num_features is automatically set by timm
        self.head = nn.Linear(self.backbone.num_features, Config.NUM_CLASSES)

    def _adapt_first_layer(self):
        """
        Identifies and replaces the first convolutional layer of the backbone.
        The new layer accepts 1 input channel, and its weights are initialized
        by averaging the weights of the original 3-channel layer.
        """
        # 1. EfficientNet Family (uses 'conv_stem')
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )
            # Initialize weights: Mean over the channel dimension (dim 1)
            # Shape: (Out, 3, K, K) -> (Out, 1, K, K)
            new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)

            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data

            # Replace the layer in the backbone
            self.backbone.conv_stem = new_layer

        # 2. ResNet Family (uses 'conv1')
        elif hasattr(self.backbone, "conv1"):
            old_layer = self.backbone.conv1
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )
            # Initialize weights: Mean over the channel dimension
            new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)

            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data

            # Replace the layer in the backbone
            self.backbone.conv1 = new_layer

        else:
            raise AttributeError(
                f"Could not identify first layer to adapt for model: {type(self.backbone)}"
            )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Feature Extraction: (B, 1, H, W) -> (B, C, H_feat, W_feat)
        x = self.backbone(x)

        # Generalized Mean Pooling: (B, C, H_feat, W_feat) -> (B, C, 1, 1)
        x = self.pooling(x)

        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)

        # Classification Head: (B, C) -> (B, 1)
        x = self.head(x)

        return x


def get_model(model_name, pretrained=True):
    """
    Factory function to instantiate the WhaleModel.

    Args:
        model_name (str): Name of the timm model (e.g., 'resnet34', 'tf_efficientnet_b0.ns_jft_in1k').
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        WhaleModel: The initialized model.
    """
    return WhaleModel(model_name, pretrained)
