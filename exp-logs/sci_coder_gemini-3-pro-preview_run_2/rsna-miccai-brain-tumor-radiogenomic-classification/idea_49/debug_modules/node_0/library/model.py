import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

logger = get_logger(name="Model")


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Dual-Scale Input Strategy.

    This model ingests a 24-channel input representing 4 modalities x 2 scales x 3 slices.
    It uses a grouped convolutional stem (groups=8) to process each 'Modality-Scale' view
    independently in the first layer, initialized with pre-trained ImageNet weights
    mapped sequentially to each group.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        logger.info(f"Initializing {Config.BACKBONE} with Asymmetric Grouped Stem...")

        # 1. Load Backbone
        # Loading with num_classes=1 to initialize dimensions, though we will rebuild the head.
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=1
        )

        # 2. Modify Stem
        self._modify_stem()

        # 3. Modify Head
        self._modify_head()

        logger.info("Model initialization complete.")

    def _modify_stem(self):
        """
        Replaces the first convolutional layer with a Grouped Convolution (groups=8).
        Performs Direct Block Copy of pre-trained weights.
        """
        old_stem = self.backbone.conv_stem

        # Ensure configuration matches expectations
        if old_stem.in_channels != 3:
            logger.warning(
                f"Expected 3 input channels in backbone stem, found {old_stem.in_channels}"
            )

        # Create new stem
        # We strictly preserve kernel_size, stride, padding, and bias settings from the original model
        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 24
            out_channels=old_stem.out_channels,  # 32
            kernel_size=old_stem.kernel_size,  # (3, 3)
            stride=old_stem.stride,  # (2, 2)
            padding=old_stem.padding,  # (1, 1)
            bias=(old_stem.bias is not None),
            groups=Config.STEM_GROUPS,  # 8
        )

        # Direct Block Copy of Weights
        # Old shape: (32, 3, 3, 3) -> (Out, In, K, K)
        # New shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K) where In/Groups = 24/8 = 3.
        # The shapes are identical, allowing direct copy.
        # This maps Filters 0-3 to Group 0 (Modality 1 Scale 1), 4-7 to Group 1 (Modality 1 Scale 2), etc.
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight)
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_stem
        logger.info(
            f"Stem modified: In={Config.IN_CHANNELS}, Groups={Config.STEM_GROUPS}, Weights Copied."
        )

    def _modify_head(self):
        """
        Reconstructs the classification head with Dropout (p=0.5).
        """
        # In timm efficientnet, the classifier is usually a Linear layer named 'classifier'.
        # We replace it to insert Dropout explicitly.

        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features

        # Create new head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
        )
        logger.info(
            f"Head modified: Dropout(p={Config.DROPOUT_RATE}) -> Linear({in_features}, 1)"
        )

    def forward(self, x):
        """
        Forward pass.
        """
        return self.backbone(x)
