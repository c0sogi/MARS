import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library import config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with 12-Channel Input.

    Adapts EfficientNet-B0 to accept 12 channels (4 modalities x 3 slices).
    Uses a grouped convolution stem (groups=4) to process each modality independently
    in the first layer while reusing the exact pre-trained ImageNet weights.
    Cite solution_lesson_node_00068: Rule of 3 (3 channels per group).
    Cite solution_lesson_node_00071: Avoid random projection in stem.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Extract Original Stem Weights
        original_conv = self.backbone.features[0][0]
        original_weights = original_conv.weight.data.clone()  # Shape: [32, 3, 3, 3]

        # 3. Construct Grouped Stem
        # Input: 12 channels. Groups: 4.
        # Each group processes 3 channels (1 modality stack).
        # Output: 32 channels (Standard EfficientNet stem output).
        # Filters per group: 32 / 4 = 8.
        # Weight shape required: [32, 3, 3, 3] (Matches original exactly!)
        self.stem_grouped = nn.Conv2d(
            in_channels=config.INPUT_CHANNELS,  # 12
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,
            bias=False,
        )

        # Normalization and Activation
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.SiLU(inplace=True)

        # Assemble the new stem block
        new_stem = nn.Sequential(self.stem_grouped, self.stem_bn, self.stem_act)

        # Replace the original stem in the backbone
        self.backbone.features[0] = new_stem

        # 4. Initialize Weights
        self.init_asymmetric_weights(original_weights)

        # 5. Modify Classifier Head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, config.NUM_CLASSES)
        )

    def init_asymmetric_weights(self, original_weights):
        """
        Initializes the grouped convolution with pre-trained ImageNet weights.
        Since we use groups=4 and input=12, the weight shape is identical to original.
        """
        with torch.no_grad():
            self.stem_grouped.weight.data = original_weights

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 24, 224, 224]

        Returns:
            torch.Tensor: Logits of shape [B, 1]
        """
        return self.backbone(x)
