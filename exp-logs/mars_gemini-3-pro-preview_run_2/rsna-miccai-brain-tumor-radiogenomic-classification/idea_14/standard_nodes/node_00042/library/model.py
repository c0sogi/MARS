import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AsymmetricEfficientNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.2, pretrained=True):
        """
        AsymmetricEfficientNet based on EfficientNet-B0.

        Args:
            num_classes (int): Number of output classes (default 1 for binary classification).
            dropout_rate (float): Dropout probability for the classifier head.
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem for 12-channel input with Grouped Convolutions
        self.modify_stem()

        # 3. Reconstruct Head with Dropout
        self.modify_head(num_classes, dropout_rate)

    def modify_stem(self):
        """
        Replaces the first convolutional layer to accept 12 channels using groups=4.
        This isolates the 4 modalities (FLAIR, T1w, T1wCE, T2w) while distributing
        pre-trained filters asymmetrically.
        """
        # Access the first Conv2d layer in the EfficientNet-B0 features block
        # features[0] is Conv2dNormActivation, index 0 is the Conv2d
        original_conv = self.backbone.features[0][0]

        # Create new stem convolution
        # Input: 12 channels (4 modalities * 3 slices)
        # Output: 32 channels (Standard B0 stem width)
        # Groups: 4 (One group per modality)
        new_conv = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,
            bias=False,
        )

        # Initialize weights
        self.initialize_asymmetric_weights(new_conv, original_conv)

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def initialize_asymmetric_weights(self, new_conv, original_conv):
        """
        Distributes the full bank of 32 pre-trained ImageNet filters across the 4 modality groups.

        Original Weight Shape: (32, 3, 3, 3) -> (Out, In, K, K)
        New Weight Shape:      (32, 3, 3, 3) -> (Out, In/Groups, K, K)

        By directly copying, we assign:
        - Filters 0-7 to Group 0 (FLAIR)
        - Filters 8-15 to Group 1 (T1w)
        - Filters 16-23 to Group 2 (T1wCE)
        - Filters 24-31 to Group 3 (T2w)
        """
        with torch.no_grad():
            if original_conv.weight is not None:
                new_conv.weight.data = original_conv.weight.data.clone()

    def modify_head(self, num_classes, dropout_rate):
        """
        Reconstructs the classifier head to strictly follow: Dropout -> Linear.
        """
        # Retrieve the input features of the original linear layer
        # In torchvision's efficientnet, classifier is a Sequential block where index 1 is Linear
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
