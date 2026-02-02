import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LearnableBatchNorm(nn.Module):
    """
    Applies Batch Normalization to the input spectrogram.
    Used to adapt audio statistics to the backbone's expected distribution.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregates temporal features using multiple parallel attention heads.
    Allows the model to focus on multiple disjoint sound events simultaneously.
    """

    def __init__(self, in_dim, num_heads=4, hidden_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

        # Project input to hidden space for attention score calculation
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        # Project hidden space to 'num_heads' attention scores
        self.linear2 = nn.Linear(hidden_dim, num_heads)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Transpose to (Batch, Time, Channels) for Linear layers
        x_t = x.transpose(1, 2)

        # Calculate attention scores
        # (Batch, Time, hidden_dim)
        attn = torch.tanh(self.linear1(x_t))
        # (Batch, Time, num_heads)
        attn = self.linear2(attn)

        # Apply softmax over the Time dimension (dim=1) to get probabilities
        attn = torch.softmax(attn, dim=1)

        # Compute weighted sum for each head
        # We want: (Batch, num_heads, Channels)
        # Transpose attn to (Batch, num_heads, Time)
        attn = attn.transpose(1, 2)

        # Batch Matrix Multiplication: (B, Heads, T) x (B, T, C) -> (B, Heads, C)
        weighted = torch.bmm(attn, x_t)

        # Flatten the heads: (Batch, num_heads * Channels)
        out = weighted.reshape(weighted.size(0), -1)

        return out


class MultiSampleDropout(nn.Module):
    """
    Applies dropout multiple times and averages the predictions.
    Acts as an internal ensemble to improve generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, in_features)
        logits = []
        for drop in self.dropouts:
            logits.append(self.linear(drop(x)))

        # Stack logits: (num_samples, Batch, out_features)
        logits = torch.stack(logits)

        # Average over samples
        return torch.mean(logits, dim=0)


class AudioClassifier(nn.Module):
    """
    Main Audio Tagging Model.
    Architecture:
    1. Learnable BatchNorm (1-channel)
    2. Input Repetition (1 -> 3 channels)
    3. ConvNeXt-Nano Backbone (ImageNet weights)
    4. Frequency Pooling
    5. Multi-Head Attention Pooling
    6. Multi-Sample Dropout Classification Head
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_heads=4,
    ):
        super().__init__()

        # 1. Input Adaptation
        # Normalize the single channel input
        self.bn0 = LearnableBatchNorm(1)

        # 2. Backbone
        # Create ConvNeXt model
        # num_classes=0 and global_pool='' ensures we get the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=""
        )

        # Determine backbone output dimension dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            # features shape: (1, C, H, W)
            self.backbone_dim = features.shape[1]

        # 3. Aggregation
        self.pool = MultiHeadAttentionPooling(
            in_dim=self.backbone_dim, num_heads=num_heads
        )

        # 4. Classification Head
        # The pooling layer outputs (Batch, num_heads * backbone_dim)
        self.head_in_dim = self.backbone_dim * num_heads
        self.fc = MultiSampleDropout(
            in_features=self.head_in_dim, out_features=num_classes
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram of shape (Batch, 1, Freq, Time)

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes)
        """
        # 1. Learnable Batch Normalization
        x = self.bn0(x)

        # 2. Input Repetition (1 -> 3 channels)
        # Repeats the channel dimension to match ImageNet backbone
        x = x.repeat(1, 3, 1, 1)

        # 3. Backbone Feature Extraction
        # Output: (Batch, C, F_down, T_down)
        x = self.backbone(x)

        # 4. Frequency Pooling
        # Collapse the frequency dimension (dim 2) to treat as a sequence over time
        # (Batch, C, F_down, T_down) -> (Batch, C, T_down)
        x = torch.mean(x, dim=2)

        # 5. Multi-Head Attention Pooling
        # Aggregates temporal sequence into a fixed vector
        # (Batch, C, T_down) -> (Batch, num_heads * C)
        x = self.pool(x)

        # 6. Classification
        logits = self.fc(x)

        return logits
