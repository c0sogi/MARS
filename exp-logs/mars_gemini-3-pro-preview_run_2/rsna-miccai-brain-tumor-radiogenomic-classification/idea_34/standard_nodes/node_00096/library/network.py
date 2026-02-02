import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class GroupedEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Fidelity-Aligned ROI Pipeline.

    This model uses EfficientNet-B0 as a backbone. The stem is surgically modified
    to accept 12 channels (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
    This enforces modality isolation in the early layers.

    Weights are initialized using a 'Direct Copy' strategy (Cite solution_lesson_node_00095).
    Since the input depth per group (3) matches the original kernel depth (3),
    we can directly reuse the pre-trained weights without interleaving.
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # Use B0 as it proved more robust for this small dataset (Cite solution_lesson_node_00095)
        weights = EfficientNet_B0_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Surgically Replace the Stem (First Convolution)
        # features[0][0] is the Conv2d layer in EfficientNet B0
        original_conv = self.backbone.features[0][0]

        # Capture original properties
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        bias = original_conv.bias

        # Create new Grouped Convolution
        # Input: 12 channels (Config.NUM_CHANNELS)
        # Groups: 4 (Config.NUM_MODALITIES) -> Each group handles 3 channels (Stack Depth)
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(bias is not None),
            groups=Config.NUM_MODALITIES,
        )

        # 3. Direct Copy Initialization (Cite solution_lesson_node_00095)
        # We copy the weights directly. Since groups=4 and in_channels=12,
        # the weight tensor shape is (Out, 12/4, K, K) = (Out, 3, K, K).
        # This matches the original weight shape exactly.
        if Config.PRETRAINED:
            with torch.no_grad():
                new_conv.weight.data = original_conv.weight.data.clone()

        # Apply replacement
        self.backbone.features[0][0] = new_conv

        # 4. Modify Classification Head
        # EfficientNet B0 classifier is Sequential[Dropout, Linear]
        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE, inplace=True),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.backbone(x)
