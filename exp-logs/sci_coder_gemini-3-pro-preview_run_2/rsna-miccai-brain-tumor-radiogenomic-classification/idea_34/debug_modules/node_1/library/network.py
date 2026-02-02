import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from library.config import Config


class GroupedEfficientNetV2(nn.Module):
    """
    Asymmetric Grouped EfficientNet-V2 with Fidelity-Aligned ROI Pipeline.

    This model uses EfficientNet-V2-S as a backbone. The stem is surgically modified
    to accept 12 channels (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
    This enforces modality isolation in the early layers.

    Weights are initialized using an 'Interleaved Asymmetric' strategy to maximize
    feature diversity across the modality groups while preserving pre-trained statistics.
    """

    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # We use V2-S as it uses Fused-MBConv, better for dense medical data
        weights = EfficientNet_V2_S_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = efficientnet_v2_s(weights=weights)

        # 2. Surgically Replace the Stem (First Convolution)
        # In EfficientNet V2, the first layer is inside features[0], which is a Conv2dNormActivation.
        # features[0][0] is the Conv2d layer.
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
        # Note: in_channels per group = 12 / 4 = 3. This matches the original RGB kernel depth (3).
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(bias is not None),
            groups=Config.NUM_MODALITIES,
        )

        # 3. Interleaved Asymmetric Initialization
        if Config.PRETRAINED:
            self._interleaved_init(new_conv, original_conv)

        # Apply replacement
        self.backbone.features[0][0] = new_conv

        # 4. Modify Classification Head
        # EfficientNet V2 classifier is a Sequential[Dropout, Linear]
        # We reconstruct it to ensure correct Dropout rate and Output size
        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE, inplace=True),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

    def _interleaved_init(self, new_conv, old_conv):
        """
        Distributes pre-trained ImageNet weights across the 4 modality groups
        using an interleaved strategy.

        Logic:
            Filter i   -> Group 1
            Filter i+1 -> Group 2
            Filter i+2 -> Group 3
            Filter i+3 -> Group 4

        This ensures every modality gets a diverse set of filters (edges, textures, etc.)
        rather than one modality getting all 'edge' filters and another getting all 'color' filters.
        """
        with torch.no_grad():
            w_old = old_conv.weight.data  # Shape: (Out, 3, K, K)
            w_new = (
                new_conv.weight.data.clone()
            )  # Shape: (Out, 3, K, K) (since 12/4 = 3)

            out_channels = w_old.shape[0]
            groups = Config.NUM_MODALITIES
            channels_per_group = out_channels // groups

            # Iterate through the filters and distribute them
            for g in range(groups):
                for i in range(channels_per_group):
                    # Destination index in the new tensor
                    # The new tensor is stacked by group: [Group1_filters, Group2_filters, ...]
                    dest_idx = g * channels_per_group + i

                    # Source index from the original tensor (Interleaved selection)
                    src_idx = i * groups + g

                    if src_idx < out_channels:
                        w_new[dest_idx] = w_old[src_idx]

            new_conv.weight.data = w_new

    def forward(self, x):
        return self.backbone(x)
