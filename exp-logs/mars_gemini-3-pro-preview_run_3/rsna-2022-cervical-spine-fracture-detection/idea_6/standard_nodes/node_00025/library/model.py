import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone for feature extraction.
    Initialized with ImageNet weights.
    The fully connected layer is replaced with Identity to return raw feature vectors.
    """

    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load pretrained ResNet18
        # Using the weights API compatible with torchvision
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.net = models.resnet18(weights=weights)

        # Replace the fully connected layer with Identity to extract features
        # ResNet18 output before FC (after avgpool) is 512 dimensions
        self.net.fc = nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of images (N, 3, H, W)
        Returns:
            torch.Tensor: Feature vectors (N, 512)
        """
        return self.net(x)


class ResidualContextBlock(nn.Module):
    """
    A Residual 1D Convolutional Block for processing sequences of slice features.
    Structure: Input + Conv1D(ReLU(BatchNorm(Input)))
    This helps smooth features across the z-axis (context modeling) while preserving
    sharp fracture signals via the residual connection.
    """

    def __init__(self, channels):
        super(ResidualContextBlock, self).__init__()
        # Pre-activation design: BN -> ReLU -> Conv
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        # Kernel size 3 to capture immediate neighbors (z-1, z, z+1 context)
        # Padding 1 to maintain sequence length
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Channels, Sequence_Length)
        Returns:
            torch.Tensor: Refined features (Batch, Channels, Sequence_Length)
        """
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return residual + out


class AnatomicallyConditionedResNet(nn.Module):
    """
    Anatomically-Conditioned Residual-Instance MIL Network.

    Architecture:
    1. Backbone: ResNet18 extracts features from each slice independently (2.5D).
    2. Positional Injection: Appends relative depth (0..1) to feature vectors to inform anatomical location.
    3. Context: Residual 1D Conv block refines features using z-axis context without blurring sparse signals.
    4. Instance Classifier: Predicts fracture probabilities for each slice independently (Instance-Level MIL).
    5. Aggregation: Global Max Pooling aggregates slice predictions to study level.
    """

    def __init__(self):
        super(AnatomicallyConditionedResNet, self).__init__()

        # 1. Backbone
        self.backbone = ResNetBackbone()
        self.feature_dim = 512  # ResNet18 output dim

        # 2. Positional Encoding
        # We append 1 dimension (relative depth) to the features
        self.combined_dim = self.feature_dim + 1

        # 3. Context Module
        self.context_block = ResidualContextBlock(self.combined_dim)

        # 4. Instance Classifier
        # Projects the context-aware features to the 7 vertebral classes
        self.classifier = nn.Linear(self.combined_dim, Config.N_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume (Batch, Slices, Channels, Height, Width)
                              Channels should be 3 (2.5D input).
        Returns:
            torch.Tensor: Study-level logits for C1-C7 (Batch, 7)
        """
        b, s, c, h, w = x.shape

        # --- Feature Extraction ---
        # Flatten batch and sequence dimensions to pass through 2D backbone
        x_flat = x.view(b * s, c, h, w)
        features = self.backbone(x_flat)  # (B*S, 512)

        # Reshape back to sequence format
        features = features.view(b, s, self.feature_dim)

        # --- Positional Injection ---
        # Generate relative depth values [0, 1]
        device = features.device
        # Create sequence [0, ..., 1]
        pos_encoding = torch.linspace(0, 1, steps=s, device=device)
        # Expand to (Batch, Slices, 1)
        pos_encoding = pos_encoding.view(1, s, 1).expand(b, s, 1)

        # Concatenate features and position
        x_seq = torch.cat([features, pos_encoding], dim=2)  # (B, S, 513)

        # --- Context Modeling ---
        # Conv1d expects (Batch, Channels, Length)
        x_seq = x_seq.permute(0, 2, 1)  # (B, 513, S)
        x_seq = self.context_block(x_seq)
        x_seq = x_seq.permute(0, 2, 1)  # (B, S, 513)

        # --- Instance Classification ---
        # Apply classifier to every slice
        # Input: (B, S, 513) -> Output: (B, S, 7)
        slice_logits = self.classifier(x_seq)

        # --- Aggregation ---
        # Global Max Pooling across the slice dimension (MIL)
        # We want the strongest fracture signal for each vertebra type across the volume
        study_logits, _ = torch.max(slice_logits, dim=1)  # (B, 7)

        return study_logits
