import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class AttentionPooling(nn.Module):
    """
    Attention Pooling module that aggregates features across the time dimension.

    This acts as a soft Multiple Instance Learning (MIL) mechanism. It learns to
    assign weights to different time steps in the feature map, allowing the model
    to focus on relevant sound events and suppress silence or background noise.
    """

    def __init__(self, in_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features

        # Attention mechanism:
        # 1. Project to hidden space (Conv1d with kernel_size=1 acts as Linear per time step)
        # 2. Apply non-linearity (Tanh)
        # 3. Project to scalar score
        self.att_conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Time)
        Returns:
            Pooled tensor of shape (Batch, Channels)
        """
        # Compute attention scores
        # attn shape: (Batch, 1, Time)
        attn = self.att_conv(x)

        # Normalize scores across time dimension so they sum to 1
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: (Batch, Channels, Time) * (Batch, 1, Time) -> Sum over Time
        # Result: (Batch, Channels)
        x = (x * attn).sum(dim=-1)

        return x


class AudioEfficientNet(nn.Module):
    """
    Audio Tagging Model based on EfficientNet-B2 with Attention Pooling.

    Architecture Pipeline:
    1. Input Adaptation: Learnable BatchNorm2d to standardize spectrograms.
    2. Backbone: EfficientNet-B2 (modified for 1-channel input) to extract features.
    3. Frequency Pooling: Average pooling to collapse frequency dimension.
    4. Time Aggregation: Attention Pooling to aggregate features over time.
    5. Classifier: Linear layer for multi-label prediction.
    """

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
    ):
        super().__init__()

        # 1. Input Adaptation
        # Learnable Batch Normalization to adapt spectrogram statistics to the backbone
        # Input shape: (Batch, 1, n_mels, Time)
        self.bn0 = nn.BatchNorm2d(1)

        # 2. Backbone
        # Load EfficientNet from timm
        # in_chans=1: Automatically modifies the first conv layer to accept 1 channel (spectrogram)
        # num_classes=0, global_pool='': Returns the spatial feature map (B, C, H, W) instead of logits
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool=""
        )

        # Determine output feature dimension dynamically
        # This ensures compatibility if the backbone model is changed
        with torch.no_grad():
            # Create dummy input: (1, 1, n_mels, 100)
            # n_mels must match the config used for spectrogram generation
            dummy = torch.randn(1, 1, CFG.n_mels, 100)
            # Pass through BN and Backbone
            dummy = self.bn0(dummy)
            features = self.backbone(dummy)
            # features shape: (1, C, H, W)
            self.n_features = features.shape[1]

        # 3. Aggregation Head (Attention Pooling)
        self.pool = AttentionPooling(self.n_features)

        # 4. Classifier
        self.fc = nn.Linear(self.n_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram (Batch, 1, n_mels, Time)
        Returns:
            Logits (Batch, num_classes)
        """
        # Adapt input statistics
        x = self.bn0(x)

        # Extract features
        # Output: (Batch, Channels, Freq_dim, Time_dim)
        x = self.backbone(x)

        # Pool over Frequency dimension (Average)
        # EfficientNet reduces spatial dims; 'height' corresponds to frequency.
        # We collapse frequency to treat the result as a sequence of time steps.
        # Shape becomes: (Batch, Channels, Time_dim)
        x = x.mean(dim=2)

        # Attention Pooling over Time
        # The model learns which time steps are important.
        # Shape becomes: (Batch, Channels)
        x = self.pool(x)

        # Classification
        # Shape becomes: (Batch, num_classes)
        x = self.fc(x)

        return x
