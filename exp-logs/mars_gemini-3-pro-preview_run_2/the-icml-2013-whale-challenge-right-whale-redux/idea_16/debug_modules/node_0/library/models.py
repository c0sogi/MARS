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

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN in pow()
        x = x.clamp(min=self.eps).pow(self.p)

        # Average pooling over spatial dimensions
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Apply inverse power
        x = x.pow(1.0 / self.p)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Right Whale Detection Model.
    Wraps a timm backbone with 1-channel adaptation and GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()

        # Create backbone without classifier and default pooling
        # This returns the feature maps directly
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Adapt the first layer to accept 1-channel input
        self._adapt_first_layer(model_name)

        # Determine the number of output features from the backbone
        # We pass a dummy input through the backbone to check shape
        # Input shape: (Batch, 1, Freq, Time)
        # Using dummy time of 100, freq of N_MELS
        dummy_input = torch.randn(2, 1, Config.N_MELS, 100)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        in_features = features.shape[1]

        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def _adapt_first_layer(self, model_name):
        """
        Replaces the first convolutional layer to accept 1 input channel.
        Weights are initialized by averaging the original RGB weights.
        """
        first_conv = None
        layer_name = None

        # Identify the first layer based on common architectures
        if "efficientnet" in model_name or "effnet" in model_name:
            first_conv = self.backbone.conv_stem
            layer_name = "conv_stem"
        elif "resnet" in model_name:
            first_conv = self.backbone.conv1
            layer_name = "conv1"
        else:
            # Generic fallback: find the first Conv2d module
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    layer_name = name
                    break

        if first_conv is None:
            raise ValueError(f"Could not find first conv layer for model: {model_name}")

        # Create new Convolutional layer
        # Keep all parameters same except in_channels
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
            dilation=first_conv.dilation,
            groups=first_conv.groups,  # Usually 1 for first layer
        )

        # Initialize weights
        # Average the weights across the channel dimension (dim 1)
        # Original weight shape: (Out, 3, K, K) -> New shape: (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(first_conv.weight, dim=1, keepdim=True)
            if first_conv.bias is not None:
                new_conv.bias[:] = first_conv.bias

        # Replace the layer in the backbone
        # We need to set the attribute on the parent module
        # Since we accessed it via self.backbone.layer_name, we set it there
        if layer_name:
            setattr(self.backbone, layer_name, new_conv)
        else:
            # Should not happen given logic above, but for safety
            raise RuntimeError("Layer name not identified for replacement.")

    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)

        # Pooling (N, C, H, W) -> (N, C, 1, 1)
        x = self.pooling(x)

        # Flatten (N, C, 1, 1) -> (N, C)
        x = x.flatten(1)

        # Classification
        x = self.fc(x)

        return x
