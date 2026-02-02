import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (AvgPool(x^p))^(1/p).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN when x is negative (though usually ReLU precedes this)
        # and to avoid division by zero issues.
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleClassifier(nn.Module):
    """
    Whale Call Classifier using a TIMM backbone with GeM pooling.
    Supports dynamic backbone selection and 1-channel input adaptation.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleClassifier, self).__init__()

        self.model_name = model_name

        # Create the backbone
        # num_classes=0 and global_pool='' ensures we get spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Adapt first convolutional layer to 1 channel
        self._adapt_input_conv()

        # Pooling layer
        self.pooling = GeM()

        # Classification Head
        # num_features is provided by timm models
        self.in_features = self.backbone.num_features
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def _adapt_input_conv(self):
        """
        Modifies the first convolutional layer to accept 1-channel input
        by averaging the weights of the original 3-channel input.
        """
        # Identify the first conv layer based on architecture
        if "efficientnet" in self.model_name:
            # EfficientNet usually uses 'conv_stem'
            old_conv = self.backbone.conv_stem
            module_name = "conv_stem"
        elif "resnet" in self.model_name:
            # ResNet usually uses 'conv1'
            old_conv = self.backbone.conv1
            module_name = "conv1"
        else:
            # Fallback: try to find the first Conv2d module
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    old_conv = module
                    module_name = name
                    break

        # Check if modification is needed
        if old_conv.in_channels == 1:
            return

        # Create new Conv2d layer with in_channels=1
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

        # Initialize weights: Average across the channel dimension (dim 1)
        # old_conv.weight shape: (Out, In, K, K) -> (Out, 3, K, K)
        # Sum across dim 1 -> (Out, K, K) -> Unsqueeze -> (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)

            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        # Replace the layer in the backbone
        # We need to set the attribute on the parent module (self.backbone)
        setattr(self.backbone, module_name, new_conv)

    def forward(self, x):
        # x shape: (B, 1, F, T)

        # Extract features
        features = self.backbone(x)  # (B, C, H, W)

        # Apply GeM Pooling
        pooled = self.pooling(features)  # (B, C, 1, 1)

        # Flatten
        flattened = pooled.view(pooled.size(0), -1)  # (B, C)

        # Classification
        logits = self.fc(flattened)  # (B, 1)

        return logits
