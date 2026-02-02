import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients for negative inputs (though usually inputs are ReLU'd)
        # Apply average pooling on x^p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Wrapper for timm models to adapt them for 1-channel input and use GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()
        self.model_name = model_name

        # Create backbone without the classification head and global pooling
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify the first convolutional layer to accept 1 channel instead of 3
        self._modify_first_conv()

        # Determine the number of output features from the backbone
        self.num_features = self._get_num_features()

        # Define the new head
        # We use GeM pooling as specified in the strategy
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def _modify_first_conv(self):
        """
        Replaces the first convolutional layer with a 1-channel version.
        Weights are initialized by averaging the original 3-channel weights.
        """

        def create_new_conv(old_conv):
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            # Initialize weights: Average across the channel dimension (dim 1)
            # Old shape: (Out, 3, K, K) -> New shape: (Out, 1, K, K)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            return new_conv

        # Handle different architectures based on naming conventions in timm
        if "resnet" in self.model_name:
            if hasattr(self.backbone, "conv1"):
                self.backbone.conv1 = create_new_conv(self.backbone.conv1)

        elif "efficientnet" in self.model_name:
            if hasattr(self.backbone, "conv_stem"):
                self.backbone.conv_stem = create_new_conv(self.backbone.conv_stem)

        elif "densenet" in self.model_name:
            # DenseNet usually has a 'features' container
            if hasattr(self.backbone, "features"):
                features = self.backbone.features
                # In timm, the first conv is often named 'conv0' inside features
                if hasattr(features, "conv0"):
                    features.conv0 = create_new_conv(features.conv0)
                elif isinstance(features, nn.Sequential):
                    # Fallback: assume first layer is the conv
                    features[0] = create_new_conv(features[0])

    def _get_num_features(self):
        """
        Computes the number of output features by running a dummy forward pass.
        """
        with torch.no_grad():
            # Create a dummy input with 1 channel
            dummy_input = torch.randn(1, 1, 128, 128)
            features = self.backbone(dummy_input)
            return features.shape[1]

    def forward(self, x):
        # Backbone extraction
        x = self.backbone(x)  # (B, C, H, W)

        # Pooling
        x = self.pooling(x)  # (B, C, 1, 1)

        # Flatten
        x = self.flatten(x)  # (B, C)

        # Classification
        x = self.fc(x)  # (B, NumClasses)

        return x


def get_model(model_name, pretrained=True):
    """
    Factory function to create the WhaleModel.

    Args:
        model_name (str): Name of the timm model (e.g., 'resnet34').
        pretrained (bool): Whether to load pretrained ImageNet/NoisyStudent weights.

    Returns:
        WhaleModel: The instantiated model.
    """
    return WhaleModel(model_name, pretrained=pretrained)
