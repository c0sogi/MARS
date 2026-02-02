import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to aggregate temporal features.
    Acts as a soft Multiple Instance Learning (MIL) mechanism, allowing the model
    to weight frames based on their relevance to the classification task.
    """

    def __init__(self, in_features):
        super(AttentionPooling, self).__init__()
        # Simple attention mechanism: Score = Wx + b
        self.attention = nn.Sequential(nn.Linear(in_features, 1), nn.Softmax(dim=1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Time, Features)
        Returns:
            Tensor of shape (Batch, Features)
        """
        # Calculate attention weights over the time dimension
        # weights: (Batch, Time, 1)
        weights = self.attention(x)

        # Compute weighted sum of features
        # (Batch, Time, Features) * (Batch, Time, 1) -> (Batch, Time, Features)
        # Sum over Time -> (Batch, Features)
        out = torch.sum(x * weights, dim=1)
        return out


class AudioClassifier(nn.Module):
    """
    Audio tagging model based on EfficientNet-B3 with Input Repetition,
    Learnable Batch Normalization, and Attention Pooling.
    """

    def __init__(self):
        super(AudioClassifier, self).__init__()

        self.num_classes = Config.NUM_CLASSES

        # 1. Input Adaptation
        # Learnable Batch Normalization to adapt spectrogram stats to ImageNet stats.
        # We expect 3 channels (either from dataset or repeated in forward).
        self.bn0 = nn.BatchNorm2d(3)

        # 2. Backbone: EfficientNet-B4
        # Initialize with ImageNet weights for transfer learning
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b4(weights=weights)

        # Extract the feature extractor (convolutional layers)
        # EfficientNet-B4 outputs 1792 channels at the final block
        self.features = self.backbone.features
        self.feat_dim = 1792

        # 3. Aggregation Head
        # Attention Pooling to aggregate temporal information
        self.att_pooling = AttentionPooling(self.feat_dim)

        # 4. Classification Head
        self.fc = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: Log-Mel Spectrogram (Batch, Channels, Freq, Time)
        Returns:
            Logits (Batch, Num_Classes)
        """
        # Logic to repeat single-channel spectrogram to 3 channels if necessary.
        # This ensures compatibility if the dataloader provides 1 channel,
        # and matches the ImageNet backbone input requirement.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Apply learnable Batch Normalization
        x = self.bn0(x)

        # Forward pass through the backbone
        # Output shape: (Batch, 1536, F', T')
        x = self.features(x)

        # Pool the Frequency dimension (Average Pooling)
        # We average over frequency to reduce dimensionality but keep time for Attention Pooling.
        # (Batch, 1536, F', T') -> (Batch, 1536, T')
        x = torch.mean(x, dim=2)

        # Permute to (Batch, T', Features) for the Attention layer
        x = x.permute(0, 2, 1)

        # Apply Attention Pooling over the Time dimension
        # (Batch, T', 1536) -> (Batch, 1536)
        x = self.att_pooling(x)

        # Final Classification
        # (Batch, 1536) -> (Batch, Num_Classes)
        x = self.fc(x)

        return x
