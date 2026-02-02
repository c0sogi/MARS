import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients with pow
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
    Whale Call Classifier using a timm backbone with GeM pooling and 1-channel adaptation.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleClassifier, self).__init__()
        self.model_name = model_name

        # 1. Load Backbone
        # Load with 3 channels initially to access original RGB weights
        # num_classes=0 removes the default classifier
        # global_pool="" returns the feature map (B, C, H, W) instead of pooled vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )

        # 2. Modify First Layer for 1-Channel Input
        self._modify_first_layer()

        # 3. Pooling Layer
        self.global_pool = GeM()

        # 4. Classification Head
        # Determine the number of output features from the backbone
        # We run a dummy forward pass to be architecture-agnostic
        with torch.no_grad():
            # Use Config.IMG_SIZE to ensure compatibility with native resolution
            dummy_input = torch.randn(1, 1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer with a 1-channel version.
        Weights are initialized by averaging the original RGB weights.
        """
        # Identify the first layer based on common timm naming conventions
        first_layer_name = None
        if hasattr(self.backbone, "conv_stem"):
            first_layer_name = "conv_stem"  # EfficientNet family
        elif hasattr(self.backbone, "conv1"):
            first_layer_name = "conv1"  # ResNet family
        else:
            # Fallback: find the first Conv2d module
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_layer_name = name
                    break

        if not first_layer_name:
            raise AttributeError(
                f"Could not find first Conv2d layer in {self.model_name}"
            )

        # Retrieve the existing layer
        old_layer = getattr(self.backbone, first_layer_name)

        # Create a new Conv2d layer with in_channels=1
        new_layer = nn.Conv2d(
            in_channels=1,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None),
        )

        # Initialize weights: Average across the channel dimension (dim 1)
        # old_layer.weight shape: (Out, 3, K, K) -> new_layer.weight shape: (Out, 1, K, K)
        with torch.no_grad():
            new_layer.weight.copy_(old_layer.weight.mean(dim=1, keepdim=True))
            if old_layer.bias is not None:
                new_layer.bias.copy_(old_layer.bias)

        # Replace the layer in the backbone
        setattr(self.backbone, first_layer_name, new_layer)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, 1, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Extract features
        x = self.backbone(x)  # (B, C, H', W')

        # Apply GeM Pooling
        x = self.global_pool(x)  # (B, C, 1, 1)

        # Flatten
        x = x.flatten(1)  # (B, C)

        # Classification
        logits = self.fc(x)  # (B, 1)

        return logits
