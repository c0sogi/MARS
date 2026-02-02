import torch
import torch.nn as nn
from torchvision import models
from library.config import NUM_CLASSES, IN_CHANNELS, STEM_GROUPS


class GroupedEfficientNet(nn.Module):
    """
    EfficientNet-B0 with a modified stem for 2.5D multi-modal input.

    This architecture is designed to handle 4 independent MRI modalities stacked depth-wise.
    By using Grouped Convolutions in the stem, we force the network to learn separate
    low-level features for each modality before mixing them in deeper layers.
    """

    def __init__(
        self, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, pretrained=True
    ):
        super(GroupedEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        # We use ImageNet weights to provide a robust starting point for feature extraction.
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify Input Stem
        # The original stem is: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # We replace it with:   Conv2d(12, 32, ..., groups=4)
        original_stem = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,
            groups=STEM_GROUPS,  # Enforce modality isolation (1 group per modality)
        )

        # 3. Robust Initialization
        # For groups=4, the weight shape is (Out, In/Groups, K, K) -> (32, 12/4, 3, 3) -> (32, 3, 3, 3).
        # This matches the original EfficientNet stem weight shape exactly.
        # We copy the pre-trained weights to preserve the learned edge/texture detectors.
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][
                    0
                ].weight.data = original_stem.weight.data.clone()

        # 4. Modify Classifier Head
        # We replace the default 1000-class head with a binary classifier.
        # We explicitly retain the Dropout layer to prevent overfitting on this small dataset.
        # The penultimate layer output size for EfficientNet-B0 is 1280.
        if hasattr(self.backbone.classifier, "1"):
            in_features = self.backbone.classifier[1].in_features
        else:
            in_features = 1280  # Default for B0

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
