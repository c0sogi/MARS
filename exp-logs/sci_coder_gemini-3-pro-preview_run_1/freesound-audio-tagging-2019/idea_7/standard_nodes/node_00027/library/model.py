import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import CFG


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Aggregates features across spatial dimensions (Time x Frequency) using a learned attention mechanism.
    This allows the model to focus on the relevant parts of the spectrogram (sound events)
    and ignore silence or noise, acting as a soft Multiple Instance Learning (MIL) pooling.
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        # A simple attention mechanism: maps feature vector to a scalar score
        # Conv1d with kernel_size=1 is equivalent to a Linear layer applied to every spatial position
        self.att_conv = nn.Conv1d(in_dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, N) where N = H * W
        x_flat = x.view(B, C, -1)

        # Calculate attention scores: (B, 1, N)
        att_scores = self.att_conv(x_flat)

        # Normalize scores across the spatial dimension to get probability distribution
        att_weights = F.softmax(att_scores, dim=2)

        # Weighted sum: (B, C)
        # Element-wise multiplication of features and weights, then sum over spatial dimension
        x_pool = torch.sum(x_flat * att_weights, dim=2)

        return x_pool


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.
    Applies multiple dropout masks to the features and averages the predictions.
    This technique acts as an internal ensemble, improving generalization and reducing overfitting.
    """

    def __init__(self, in_features, out_features, num_samples=5, drop_rate=0.5):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (B, in_features)
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout and then the classifier
            logits_list.append(self.fc(dropout(x)))

        # Stack predictions and calculate mean: (B, out_features)
        return torch.stack(logits_list, dim=0).mean(dim=0)


class AudioResNeSt(nn.Module):
    """
    Audio Tagging Model based on ResNeSt-50d.

    Architecture:
    1. Learnable Batch Normalization: Adapts spectrogram statistics.
    2. Input Repetition: Expands 1-channel audio to 3-channel for backbone compatibility.
    3. ResNeSt-50d Backbone: Extracts strong spectral features using Split-Attention.
    4. Attention Pooling: Aggregates features focusing on active sound regions.
    5. Multi-Sample Dropout: Robust classification head.
    """

    def __init__(self, pretrained=True):
        super(AudioResNeSt, self).__init__()

        # 1. Learnable Batch Normalization
        # Input is (B, 1, F, T). Normalizes the single-channel input spectrogram.
        self.bn0 = nn.BatchNorm2d(1)

        # 2. Backbone
        # Create ResNeSt-50d model.
        # num_classes=0 removes the original FC layer.
        # global_pool="" removes the original pooling, returning (B, C, H, W) features.
        self.encoder = timm.create_model(
            CFG.model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of output channels from the encoder
        self.in_features = self.encoder.num_features

        # 3. Attention Pooling
        self.att_pooling = AttentionPooling(self.in_features)

        # 4. Multi-Sample Dropout Head
        self.head = MultiSampleDropout(
            in_features=self.in_features,
            out_features=CFG.num_classes,
            num_samples=5,
            drop_rate=0.5,
        )

    def forward(self, x):
        # Input x shape: (B, 1, F, T)

        # Apply Learnable Batch Normalization
        x = self.bn0(x)

        # Input Repetition: (B, 1, F, T) -> (B, 3, F, T)
        # Repeats the mono spectrogram to satisfy the RGB input requirement of the backbone
        x = x.repeat(1, 3, 1, 1)

        # Feature Extraction
        # Output shape: (B, C, H, W)
        x = self.encoder(x)

        # Attention Pooling
        # Output shape: (B, C)
        x = self.att_pooling(x)

        # Classification
        # Output shape: (B, num_classes)
        x = self.head(x)

        return x
