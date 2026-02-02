import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0.

    Features:
    1. Grouped Convolutional Stem (groups=4) to isolate 4 MRI modalities (FLAIR, T1w, T1wCE, T2w).
    2. Asymmetric Filter Initialization: Distributes the 32 pre-trained ImageNet filters
       across the 4 groups to maximize feature diversity.
    3. Regularized Binary Classification Head.
    """

    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.2):
        super(AsymmetricEfficientNet, self).__init__()

        # Load the backbone with pre-trained weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # ----------------------------------------------------------------------
        # 1. Modify Stem: Grouped Convolution & Asymmetric Init
        # ----------------------------------------------------------------------
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_stem = self.model.features[0][0]

        # New Stem:
        # - Input: 12 channels (4 modalities * 3 slices)
        # - Output: 32 channels (same as original)
        # - Groups: 4 (One group per modality)
        # - Kernel/Stride/Padding: Same as original
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=(original_stem.bias is not None),
            groups=4,
        )

        # Apply Asymmetric Initialization
        self._init_asymmetric_weights(original_stem, new_stem)

        # Replace the layer in the backbone
        self.model.features[0][0] = new_stem

        # ----------------------------------------------------------------------
        # 2. Modify Head: Regularized Binary Classifier
        # ----------------------------------------------------------------------
        # EfficientNet-B0 classifier structure:
        # Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )

        # Retrieve the input features size from the existing linear layer
        in_features = self.model.classifier[1].in_features

        # Reconstruct the classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def _init_asymmetric_weights(self, original_layer, new_layer):
        """
        Distributes the pre-trained filters from the original RGB stem across the
        4 groups of the new stem.

        Logic:
        - Original Weight Shape: (32, 3, 3, 3) -> (Out, In, K, K)
        - New Weight Shape:      (32, 3, 3, 3) -> (Out, In/Groups, K, K)

        With groups=4:
        - Filters 0-7   process Input Channels 0-2   (Modality 1)
        - Filters 8-15  process Input Channels 3-5   (Modality 2)
        - Filters 16-23 process Input Channels 6-8   (Modality 3)
        - Filters 24-31 process Input Channels 9-11  (Modality 4)

        By copying the weights directly, we assign a unique subset of the original
        ImageNet filters to each modality. This preserves the full diversity of
        feature extractors (edges, textures) learned on ImageNet, rather than
        replicating the same small subset for all modalities.
        """
        with torch.no_grad():
            new_layer.weight.data.copy_(original_layer.weight.data)
            if original_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.data.copy_(original_layer.bias.data)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.model(x)
