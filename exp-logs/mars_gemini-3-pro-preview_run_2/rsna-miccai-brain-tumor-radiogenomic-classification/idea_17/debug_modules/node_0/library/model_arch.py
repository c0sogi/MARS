import torch
import torch.nn as nn
import timm
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    This model modifies the standard EfficientNet-B0 to accept 12-channel inputs
    (4 modalities x 3 slices) via a Grouped Convolutional Stem. It utilizes
    Asymmetric Filter Initialization to distribute pre-trained ImageNet filters
    across the modalities, and includes a regularized classification head.
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pretrained Backbone
        # We load the model with the target number of classes, but we will
        # rebuild the head anyway to ensure the specific Dropout structure.
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=True, num_classes=Config.NUM_CLASSES
        )

        # ----------------------------------------------------------------------
        # 2. Modify Stem (Grouped Convolution & Asymmetric Init)
        # ----------------------------------------------------------------------
        # Original stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # We want: Conv2d(12, 32, ..., groups=4)

        original_stem = self.backbone.conv_stem

        # Extract parameters from original stem
        out_channels = original_stem.out_channels
        kernel_size = original_stem.kernel_size
        stride = original_stem.stride
        padding = original_stem.padding
        bias = original_stem.bias is not None

        # Create new stem with grouped convolutions
        # Config.IN_CHANNELS = 12, Config.STEM_GROUPS = 4
        self.new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=Config.STEM_GROUPS,
        )

        # Asymmetric Filter Initialization
        # The weight shape for Conv2d is (Out, In/Groups, K, K).
        # Original (Groups=1): (32, 3, 3, 3)
        # New (Groups=4): (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        # Since shapes match, we copy weights directly. This assigns:
        # Filters 0-7 -> Modality 1 (Channels 0-2)
        # Filters 8-15 -> Modality 2 (Channels 3-5)
        # etc.
        with torch.no_grad():
            self.new_stem.weight.copy_(original_stem.weight)
            if bias:
                self.new_stem.bias.copy_(original_stem.bias)

        # Replace the stem in the backbone
        self.backbone.conv_stem = self.new_stem

        # ----------------------------------------------------------------------
        # 3. Modify Head (Regularization)
        # ----------------------------------------------------------------------
        # Explicitly reconstruct the head to include Dropout -> Linear

        # Identify the classifier layer (usually named 'classifier' in EfficientNet)
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROP_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )
        elif hasattr(self.backbone, "fc"):
            # Fallback for some timm models that use 'fc'
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=Config.DROP_RATE),
                nn.Linear(in_features, Config.NUM_CLASSES),
            )
        else:
            raise AttributeError("Could not locate classifier layer in backbone.")

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
