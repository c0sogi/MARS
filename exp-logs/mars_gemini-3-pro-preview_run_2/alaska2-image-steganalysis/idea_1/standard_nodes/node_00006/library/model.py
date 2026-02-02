import torch
import torch.nn as nn
import timm
from library.utils import get_srm_weights


class SRMConv(nn.Module):
    """
    A fixed convolutional layer initialized with Spatial Rich Model (SRM) filters.
    This layer acts as a residual extractor, suppressing image content and exposing noise.
    """

    def __init__(self):
        super(SRMConv, self).__init__()
        # The SRM filters are 5x5. We have 30 of them.
        # We define a Conv2d layer that takes 3 input channels (RGB) and produces 30 output channels.
        # Padding is set to 2 to preserve spatial dimensions (5x5 kernel).
        self.srm_layer = nn.Conv2d(
            3, 30, kernel_size=5, padding=2, stride=1, bias=False
        )

        # Load the 30 SRM filters from the utility function.
        # Shape: (30, 1, 5, 5)
        srm_weights = get_srm_weights()

        # Adapt weights for RGB input.
        # We repeat the (30, 1, 5, 5) weights 3 times along the input channel dimension.
        # This means each of the 30 filters is applied to R, G, and B channels and the results are summed.
        # Shape becomes: (30, 3, 5, 5)
        self.srm_layer.weight.data = srm_weights.repeat(1, 3, 1, 1)

        # Freeze the parameters so they are not updated during training.
        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).
        Returns:
            torch.Tensor: Residual maps of shape (Batch, 30, Height, Width).
        """
        return self.srm_layer(x)


class SRMEfficientNet(nn.Module):
    """
    Baseline Steganalysis Model.
    Architecture:
        1. SRMConv (Fixed Stem): Extracts noise residuals.
        2. Projection (Trainable): Maps 30 residual channels to 3 channels.
        3. EfficientNet-B0 (Backbone): Extracts high-level features and classifies.
    """

    def __init__(self, model_name, pretrained=True):
        super(SRMEfficientNet, self).__init__()

        # 1. Fixed SRM Stem
        self.srm_conv = SRMConv()

        # 2. Feature Adaptation / Projection Layer
        # Projects the 30-channel residuals into 3 channels to be compatible with the backbone.
        # We use a 3x3 convolution to allow for local spatial integration of residuals.
        # BatchNorm and Activation are added to stabilize input distribution for the pre-trained backbone.
        self.projection = nn.Sequential(
            nn.Conv2d(30, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),  # SiLU (Swish) is commonly used with EfficientNets
        )

        # 3. Backbone
        # Load EfficientNet from timm.
        # num_classes=1 configures the final linear layer for binary classification.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).
        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Step 1: Extract Residuals
        # Input: (B, 3, H, W) -> Output: (B, 30, H, W)
        x = self.srm_conv(x)

        # Step 2: Project to 3 channels
        # Input: (B, 30, H, W) -> Output: (B, 3, H, W)
        x = self.projection(x)

        # Step 3: Backbone Feature Extraction & Classification
        # Input: (B, 3, H, W) -> Output: (B, 1)
        x = self.backbone(x)

        return x
