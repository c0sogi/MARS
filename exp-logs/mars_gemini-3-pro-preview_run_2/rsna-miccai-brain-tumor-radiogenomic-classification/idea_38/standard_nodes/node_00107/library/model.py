import torch
import torch.nn as nn
import timm
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Strict Geometric Alignment support.

    Architecture:
    - Backbone: EfficientNet-B0 (pretrained)
    - Stem: Grouped Convolution (groups=4) for 4 modalities (FLAIR, T1w, T1wCE, T2w).
            Input channels: 12 (4 modalities * 3 slices).
    - Initialization: Direct Asymmetric Initialization (Direct Block Copy).
    - Head: Dropout (0.5) + Linear Projection.
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()
        # 1. Load Pretrained Backbone
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)

        # 2. Modify Stem for Multi-Modality Grouped Processing
        self._modify_stem()

        # 3. Modify Head for Regularization and Binary Classification
        self._modify_head()

    def _modify_stem(self):
        """
        Replaces the first convolutional layer to accept 12 channels with groups=4.
        """
        original_stem = self.backbone.conv_stem

        # Define new stem
        # Input: 12 channels (4 modalities x 3 slices)
        # Output: 32 channels (standard EfficientNet-B0 stem output)
        # Groups: 4 (One group per modality)
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,  # Original stem has no bias
            groups=4,
        )

        # Initialize weights
        self._init_weights(original_stem, new_stem)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_stem

    def _init_weights(self, original_stem, new_stem):
        """
        Performs Direct Asymmetric Initialization.
        Copies weights from the original RGB stem to the new grouped stem.

        Original Weight Shape: (32, 3, 3, 3) -> [Out, In, K, K]
        New Weight Shape:      (32, 3, 3, 3) -> [Out, In/Groups, K, K] (12/4 = 3)

        Logic:
        Directly copy the tensor.
        - Filters 0-7 (originally RGB) -> Assigned to Group 0 (FLAIR)
        - Filters 8-15 (originally RGB) -> Assigned to Group 1 (T1w)
        - ... and so on.
        This avoids interleaving and preserves filter coherence within groups.
        """
        with torch.no_grad():
            new_stem.weight.data = original_stem.weight.data.clone()

    def _modify_head(self):
        """
        Replaces the classifier head with Dropout and a single Linear layer.
        """
        # Get input features of the original classifier
        in_features = self.backbone.classifier.in_features

        # Replace with regularized head
        # Cite Lesson 00087: Aggressive regularization for small datasets
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
