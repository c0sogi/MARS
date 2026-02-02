import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean: f(X) = (1/N * sum(x^p))^(1/p)
    where p is a learnable parameter.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp min to eps to avoid numerical instability with pow
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


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for Right Whale Detection.

    Architecture:
    1. Backbone: tf_efficientnet_b0.ns_jft_in1k (Noisy Student weights)
    2. Input Adaptation: First Conv layer modified for 1-channel input (averaged weights)
    3. Pooling: Generalized Mean Pooling (GeM)
    4. Head: Linear Classification Layer
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(WhaleEfficientNet, self).__init__()

        # Create backbone using timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Determine the number of input features for the head
        if hasattr(self.backbone, "num_features"):
            n_features = self.backbone.num_features
        elif hasattr(self.backbone, "classifier"):
            n_features = self.backbone.classifier.in_features
        else:
            # Fallback for some timm models
            n_features = self.backbone.fc.in_features

        # Modify the first layer to accept 1 channel instead of 3
        self._modify_first_layer()

        # Remove original head and pooling to save computation/memory
        # (Though forward_features skips them, it's good practice)
        if hasattr(self.backbone, "global_pool"):
            self.backbone.global_pool = nn.Identity()
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        # Define Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Define Classification Head
        self.head = nn.Linear(n_features, Config.NUM_CLASSES)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 1-channel version.
        Weights are initialized by averaging the original RGB weights to preserve spatial filters.
        """
        # EfficientNet uses 'conv_stem' for the first layer
        if not hasattr(self.backbone, "conv_stem"):
            raise AttributeError(
                f"Backbone {Config.MODEL_NAME} does not have 'conv_stem'."
            )

        old_layer = self.backbone.conv_stem

        # Create new Conv2d layer
        new_layer = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 1
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=old_layer.bias is not None,
        )

        # Initialize weights
        # old_layer.weight: (Out, 3, H, W)
        # new_layer.weight: (Out, 1, H, W)
        with torch.no_grad():
            new_layer.weight[:] = torch.mean(old_layer.weight, dim=1, keepdim=True)
            if old_layer.bias is not None:
                new_layer.bias[:] = old_layer.bias

        self.backbone.conv_stem = new_layer

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, 1, F, T)
        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        # 1. Feature Extraction
        # forward_features returns the feature map (B, C, H', W')
        x = self.backbone.forward_features(x)

        # 2. Pooling
        # Transforms (B, C, H', W') -> (B, C, 1, 1)
        x = self.pooling(x)

        # 3. Flatten
        # Transforms (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)

        # 4. Classification Head
        # Transforms (B, C) -> (B, NUM_CLASSES)
        x = self.head(x)

        return x
