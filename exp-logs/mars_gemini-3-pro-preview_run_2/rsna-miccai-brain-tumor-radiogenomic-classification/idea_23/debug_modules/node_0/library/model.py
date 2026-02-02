import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric EfficientNet-B0 for MGMT Promoter Methylation Prediction.

    This model implements the 'Asymmetric Grouped EfficientNet' strategy:
    1.  **Grouped Stem**: The input layer is modified to accept 12 channels (4 modalities * 3 slices)
        using grouped convolutions (groups=4). This ensures that each modality (FLAIR, T1w, T1wCE, T2w)
        is processed independently in the first layer.
    2.  **Asymmetric Initialization**: The 32 pre-trained ImageNet filters are distributed across
        the 4 modality groups. This preserves the full diversity of edge and texture detectors
        learned from ImageNet, applying distinct subsets of filters to each modality.
    3.  **Regularized Head**: The classifier is reconstructed with Dropout and a single Linear projection.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use IMAGENET1K_V1 weights to leverage transfer learning
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify Stem (First Convolutional Layer)
        # Original stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_stem = self.backbone.features[0][0]

        # Ensure configuration matches the architectural expectation
        if Config.IN_CHANNELS != 12:
            raise ValueError(
                f"AsymmetricEfficientNet expects 12 input channels, but Config.IN_CHANNELS is {Config.IN_CHANNELS}"
            )

        # Create the new stem with Grouped Convolutions
        # groups=4 splits the 12 inputs into 4 groups of 3 (one group per modality).
        # The 32 output filters are also split into 4 groups of 8.
        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
            groups=4,
        )

        # 3. Asymmetric Filter Distribution Initialization
        # The original weights have shape [32, 3, 3, 3] (Out, In, K, K).
        # The new weights (with groups=4) effectively have shape [32, 3, 3, 3] in PyTorch's internal representation
        # because PyTorch stores weights as [out_channels, in_channels/groups, k, k].
        #
        # By copying the weights directly, we assign filters 0-7 to Group 0 (Modality 1),
        # filters 8-15 to Group 1 (Modality 2), and so on. This distributes the pre-trained
        # diversity across modalities rather than replicating the same filters for all.
        with torch.no_grad():
            new_stem.weight.copy_(original_stem.weight)

        # Replace the stem in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Reconstruct Classifier Head
        # EfficientNet's classifier is a Sequential block. We replace it to ensure
        # we have the correct output dimension (1 class) and regularization structure.
        original_classifier = self.backbone.classifier

        # Extract the input features (1280 for B0)
        in_features = original_classifier[-1].in_features

        # Extract dropout probability (default to 0.2 if not explicitly found)
        dropout_p = 0.2
        if len(original_classifier) > 0 and isinstance(
            original_classifier[0], nn.Dropout
        ):
            dropout_p = original_classifier[0].p

        # Define the new head: Dropout -> Linear
        # We output 1 value (logit) for binary classification.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
