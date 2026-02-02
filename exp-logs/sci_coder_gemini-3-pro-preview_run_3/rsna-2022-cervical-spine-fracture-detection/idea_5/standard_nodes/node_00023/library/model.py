import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ResidualContextBlock(nn.Module):
    """
    Residual 1D Convolutional Block for context modeling.
    Structure: F_out = F_in + Conv1D(ReLU(Conv1D(F_in)))

    This block allows the model to integrate information from neighboring slices
    while the residual connection preserves the sharp, high-frequency signal
    of a fracture from the original slice features.
    """

    def __init__(self, channels):
        super(ResidualContextBlock, self).__init__()
        # Preserve sequence length: kernel_size=3, padding=1
        self.conv1 = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Channels, Seq_Len)
        Returns:
            Tensor of shape (Batch, Channels, Seq_Len)
        """
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class CervicalFractureModel(nn.Module):
    """
    Anatomically-Conditioned Residual Sequence MIL Model.

    Architecture:
    1. Backbone: ResNet34 (2.5D input, slice-wise feature extraction).
    2. Positional Injection: Concatenate relative depth (z/N) to feature vectors.
    3. Context: Residual 1D Convolution over the sequence dimension.
    4. Aggregation: Global Max Pooling to detect sparse anomalies.
    5. Head: Classification for 7 cervical vertebrae.
    """

    def __init__(self, pretrained=True):
        super(CervicalFractureModel, self).__init__()

        # --- Backbone ---
        # Using ResNet18 to allow for larger batch size (stability)
        # Cite solution_lesson_node_00022
        # Note: Input is 3 channels (2.5D stacking), which matches standard ResNet input.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Feature dimension for ResNet18 is 512
        self.feature_dim = 512

        # Isolate the feature extractor (up to layer4)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Slice-wise pooling to get a feature vector per slice
        self.slice_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Context Modeling ---
        # Input dimension = Feature Dim + 1 (Positional Encoding)
        self.context_channels = self.feature_dim + 1
        self.context_block = ResidualContextBlock(self.context_channels)

        # --- Classification Head ---
        # Output: 7 logits (C1-C7)
        self.classifier = nn.Linear(self.context_channels, Config.NUM_CLASSES)

    def forward(self, images, positions):
        """
        Args:
            images: (Batch, Seq_Len, C, H, W) - 2.5D Slice Stacks
            positions: (Batch, Seq_Len, 1) - Normalized relative depth

        Returns:
            logits: (Batch, Num_Classes)
        """
        batch_size, seq_len, c, h, w = images.shape

        # 1. Feature Extraction (Slice-wise)
        # Flatten batch and sequence dimensions to process slices in parallel
        cnn_input = images.view(batch_size * seq_len, c, h, w)

        # Pass through backbone
        features = self.backbone(cnn_input)  # Shape: (B*S, 512, H', W')
        features = self.slice_pool(features)  # Shape: (B*S, 512, 1, 1)
        features = features.flatten(1)  # Shape: (B*S, 512)

        # Reshape back to sequence format
        features = features.view(
            batch_size, seq_len, self.feature_dim
        )  # Shape: (B, S, 512)

        # 2. Positional Injection
        # positions is (B, S, 1). Concatenate along the feature dimension.
        # This explicitly informs the model of the vertebral level (e.g., C1 vs C7).
        features = torch.cat([features, positions], dim=2)  # Shape: (B, S, 513)

        # 3. Residual 1D Context Block
        # Conv1D expects (Batch, Channels, Length)
        features = features.permute(0, 2, 1)  # Shape: (B, 513, S)
        features = self.context_block(features)

        # 4. Aggregation (Global Max Pooling)
        # Max pool over the sequence dimension (dim 2)
        # We use Max Pooling to detect the presence of a fracture (anomaly) anywhere in the sequence.
        features = torch.max(features, dim=2)[0]  # Shape: (B, 513)

        # 5. Classification
        logits = self.classifier(features)  # Shape: (B, 7)

        return logits
