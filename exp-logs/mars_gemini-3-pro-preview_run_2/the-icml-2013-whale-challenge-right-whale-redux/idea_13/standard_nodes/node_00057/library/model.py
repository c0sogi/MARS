import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.

    Formula: f(X) = (1/N * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small constant to avoid numerical instability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp min value to eps to avoid NaNs when raising to power p
        # Average pooling is applied over the spatial dimensions (H, W)
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleClassifier(nn.Module):
    """
    Right Whale Detection Classifier.
    Wraps a timm backbone with GeM pooling and a custom classification head.
    Adapts 3-channel pre-trained weights to 1-channel input via weight averaging.
    """

    def __init__(self, model_name, pretrained=True, in_chans=1, num_classes=1):
        """
        Args:
            model_name (str): Name of the architecture to load from timm (e.g., 'resnet34').
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            in_chans (int): Number of input channels (1 for spectrograms).
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super(WhaleClassifier, self).__init__()

        # Create backbone
        # num_classes=0 and global_pool="" ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )

        # Enforce "Weight Averaging" for 1-channel adaptation
        # timm's default behavior for in_chans=1 is to sum the weights of the original 3 channels.
        # To achieve averaging (Mean), we must divide the weights of the first layer by 3.
        if in_chans == 1:
            self._scale_first_conv()

        # Feature dimension is usually available as num_features
        self.num_features = self.backbone.num_features

        # Pooling Layer
        self.pool = GeM()

        # Classification Head
        self.fc = nn.Linear(self.num_features, num_classes)

    def _scale_first_conv(self):
        """
        Scales the weights of the first convolutional layer by 1/3.
        This converts the 'sum' adaptation performed by timm into an 'average' adaptation.
        """
        first_conv = None

        # Identify the first conv layer based on common architecture naming conventions
        if hasattr(self.backbone, "conv_stem"):
            first_conv = self.backbone.conv_stem  # EfficientNet, etc.
        elif hasattr(self.backbone, "conv1"):
            first_conv = self.backbone.conv1  # ResNet, etc.
        else:
            # Fallback: Search for the first Conv2d module
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    break

        if first_conv is not None:
            with torch.no_grad():
                # Divide weights by 3.0 to average the original RGB weights
                first_conv.weight.mul_(1.0 / 3.0)

    def forward(self, x):
        # 1. Feature Extraction
        # Output shape: (B, C, H, W)
        x = self.backbone(x)

        # 2. GeM Pooling
        # Output shape: (B, C, 1, 1)
        x = self.pool(x)

        # 3. Flatten
        # Output shape: (B, C)
        x = x.view(x.size(0), -1)

        # 4. Classification
        # Output shape: (B, num_classes)
        x = self.fc(x)

        return x
