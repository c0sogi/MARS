import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import (
    BACKBONE,
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    BIDIRECTIONAL,
    TARGET_COLS,
    IN_CHANNELS,
)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer to aggregate a sequence of features.
    Computes a weighted sum of the sequence elements based on learned attention scores.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Seq_Len, Features).

        Returns:
            torch.Tensor: Aggregated feature vector of shape (Batch, Features).
        """
        # Compute attention scores: (Batch, Seq_Len, 1)
        scores = self.attention_net(x)

        # Normalize scores to weights: (Batch, Seq_Len, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (Batch, Features)
        context = torch.sum(x * weights, dim=1)
        return context


class CervicalSpineSeqModel(nn.Module):
    """
    Sequential 2.5D Multiple Instance Learning Model for Cervical Spine Fracture Detection.

    Architecture:
    1. 2.5D Input (3 slices: z-1, z, z+1)
    2. CNN Backbone (Feature Extractor)
    3. Bidirectional LSTM (Anatomical Context)
    4. Attention Pooling (Study-level Aggregation)
    5. Classification Head
    """

    def __init__(self, pretrained=True):
        super(CervicalSpineSeqModel, self).__init__()

        # 1. CNN Backbone
        # We use timm to create the backbone. num_classes=0 removes the final FC layer
        # and returns the pooled features (or flattened features depending on model).
        self.backbone = timm.create_model(
            BACKBONE, pretrained=pretrained, num_classes=0, in_chans=IN_CHANNELS
        )

        # Determine the output dimension of the backbone
        if hasattr(self.backbone, "num_features"):
            self.backbone_dim = self.backbone.num_features
        else:
            # Fallback for some models, though num_features is standard in timm
            with torch.no_grad():
                dummy = torch.zeros(1, IN_CHANNELS, 224, 224)
                out = self.backbone(dummy)
                self.backbone_dim = out.shape[1]

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.backbone_dim,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
        )

        # Calculate LSTM output dimension
        self.lstm_out_dim = LSTM_HIDDEN_DIM * (2 if BIDIRECTIONAL else 1)

        # 3. Attention Pooling
        self.attention = AttentionPooling(self.lstm_out_dim)

        # 4. Classification Head
        self.classifier = nn.Linear(self.lstm_out_dim, len(TARGET_COLS))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Channels, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        b, s, c, h, w = x.shape

        # Flatten Batch and Sequence dimensions to process slices in parallel
        # Shape: (Batch * Seq_Len, Channels, H, W)
        x = x.view(b * s, c, h, w)

        # Extract features using CNN backbone
        # Shape: (Batch * Seq_Len, Backbone_Dim)
        features = self.backbone(x)

        # Reshape back to sequence format
        # Shape: (Batch, Seq_Len, Backbone_Dim)
        features = features.view(b, s, -1)

        # Pass through LSTM
        # Shape: (Batch, Seq_Len, LSTM_Out_Dim)
        lstm_out, _ = self.lstm(features)

        # Aggregate sequence using Attention Pooling
        # Shape: (Batch, LSTM_Out_Dim)
        pooled_embedding = self.attention(lstm_out)

        # Classification
        # Shape: (Batch, Num_Classes)
        logits = self.classifier(pooled_embedding)

        return logits
