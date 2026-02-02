import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("Model")


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    Implements Idea 37:
    - Backbone: EfficientNet-B0 (ImageNet weights)
    - Stem: 12-channel Grouped Conv (groups=4) to isolate modalities.
    - Init: Direct Block Copy of weights from RGB stem.
    - Head: Regularized with Dropout(p=0.5).
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        logger.info(f"Initializing {Config.MODEL_NAME} with Asymmetric Grouped Stem...")

        # 1. Load Backbone with Pre-trained Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem (Grouped Convolution)
        # The stem in EfficientNet-B0 is the first layer of the 'features' sequential block.
        # Structure: features[0] -> Conv2dNormActivation -> [0]: Conv2d, [1]: BN, [2]: SiLU
        old_conv = self.backbone.features[0][0]

        # Create new stem
        # We maintain kernel, stride, and padding to preserve geometry.
        # We set groups=4 to isolate the 4 modalities (FLAIR, T1w, T1wCE, T2w).
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            groups=Config.STEM_GROUPS,
            bias=old_conv.bias is not None,
        )

        # 3. Initialize Weights (Direct Block Copy)
        self.initialize_weights(new_conv, old_conv)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Modify Head (Regularized)
        # Replace the classifier to output 1 class (binary) and increase dropout.
        # Original: Sequential(Dropout(p=0.2), Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=1, bias=True),
        )

        logger.info("Model initialization complete.")

    def initialize_weights(self, new_conv: nn.Conv2d, old_conv: nn.Conv2d):
        """
        Performs 'Direct Block Copy' of weights.

        The original stem has shape (32, 3, 3, 3) [Out, In, K, K].
        The new stem has shape (32, 3, 3, 3) [Out, In/Groups, K, K] because 12 / 4 = 3.

        We copy the weights directly. This creates an asymmetric assignment:
        - Filters 0-7  (trained on RGB) -> Assigned to Group 1 (FLAIR)
        - Filters 8-15 (trained on RGB) -> Assigned to Group 2 (T1w)
        - ... and so on.

        We explicitly avoid interleaving to preserve the internal correlation
        of the pre-trained filter banks.
        """
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

        logger.info("Weights initialized via Direct Block Copy.")

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
