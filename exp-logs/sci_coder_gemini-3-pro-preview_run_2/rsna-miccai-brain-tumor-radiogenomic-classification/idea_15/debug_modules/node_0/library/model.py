import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AsymmetricGroupedEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Modality-Specific Normalization.

    This architecture modifies the standard EfficientNet-B0 to process 4 independent
    MRI modalities (FLAIR, T1w, T1wCE, T2w) simultaneously while maintaining
    feature isolation in the early stages.

    Key Modifications:
    1. Stem Convolution: Replaced with Grouped Convolution (groups=4) taking 12 input channels.
       - Weights are initialized by distributing the 32 pre-trained ImageNet filters
         across the 4 modality groups (Asymmetric Initialization).
    2. Stem Normalization: Batch Normalization is replaced with Group Normalization (groups=4).
       - This ensures that intensity statistics are normalized per-modality rather than
         globally or per-batch, preserving the distinct contrast profiles of each MRI sequence.
    3. Head: Custom classifier with Dropout and a single output unit.
    """

    def __init__(self):
        super(AsymmetricGroupedEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # Use ImageNet weights to leverage learned texture/edge detectors
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # ----------------------------------------------------------------------
        # 2. Modify Stem Convolution (Modality Isolation)
        # ----------------------------------------------------------------------
        # The first layer of EfficientNet-B0 features is a Conv2dNormActivation block.
        # features[0][0] is the Conv2d layer.
        original_conv = self.backbone.features[0][0]

        # Create a new Conv2d layer with:
        # - in_channels=12 (4 modalities * 3 slices)
        # - groups=4 (To isolate each modality into its own set of filters)
        # - out_channels=32 (Same as original)
        new_conv = nn.Conv2d(
            in_channels=Config.TOTAL_INPUT_CHANNELS,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
            groups=Config.CONV_GROUPS,
        )

        # Asymmetric Filter Initialization:
        # The original weights have shape (32, 3, 3, 3) -> (Out, In, K, K).
        # The new weights, due to grouping, have shape (32, 12//4, 3, 3) -> (32, 3, 3, 3).
        # Since the shapes are identical, we can copy the weights directly.
        # Because of the grouped connectivity:
        # - Filters 0-7 will process Input Channels 0-2 (Modality 1)
        # - Filters 8-15 will process Input Channels 3-5 (Modality 2)
        # - etc.
        # This effectively distributes the diverse bank of ImageNet filters across modalities.
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight)

        # Replace the original convolution
        self.backbone.features[0][0] = new_conv

        # ----------------------------------------------------------------------
        # 3. Modify Normalization (Modality-Specific Norm)
        # ----------------------------------------------------------------------
        # features[0][1] is the BatchNorm2d layer.
        original_bn = self.backbone.features[0][1]

        # Replace with Group Normalization.
        # num_groups=4 ensures that normalization statistics (mean, std) are computed
        # independently for each modality's resulting feature maps.
        new_gn = nn.GroupNorm(
            num_groups=Config.NORM_GROUPS, num_channels=original_bn.num_features
        )

        # Initialize GroupNorm affine parameters from the original BatchNorm
        # to preserve the scale of the pre-trained features.
        with torch.no_grad():
            if original_bn.weight is not None:
                new_gn.weight.copy_(original_bn.weight)
            if original_bn.bias is not None:
                new_gn.bias.copy_(original_bn.bias)

        # Replace the batch norm layer
        self.backbone.features[0][1] = new_gn

        # ----------------------------------------------------------------------
        # 4. Modify Classifier Head
        # ----------------------------------------------------------------------
        # Retrieve the input features size for the final linear layer (typically 1280)
        in_features = self.backbone.classifier[1].in_features

        # Rebuild classifier with specified Dropout and Binary Output
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE, inplace=True),
            nn.Linear(in_features=in_features, out_features=Config.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
