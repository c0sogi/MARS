import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.checkpoint import checkpoint
from library.config import Config


class SoftSpatialAttention(nn.Module):
    """
    Learnable Soft Spatial Attention module.
    Predicts a spatial mask (1 channel) from input features and performs
    weighted spatial pooling.
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Convolution to collapse channels to 1 attention map
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Height, Width)

        # Generate attention map: (B, 1, H, W)
        att_map = torch.sigmoid(self.conv(x))

        # Apply attention weighting
        # Broadcast multiply: (B, C, H, W) * (B, 1, H, W)
        x = x * att_map

        # Sum over spatial dimensions to pool
        # Output: (B, C)
        # Cast to float32 to prevent FP16 overflow during summation
        x = x.float().sum(dim=(2, 3))

        return x


class MultiScaleBackbone(nn.Module):
    """
    EfficientNet-B4 backbone extracting and fusing P4 and P5 features.
    """

    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        # Load model with features_only=True to get intermediate feature maps
        # out_indices=(3, 4) corresponds to P4 (stride 16) and P5 (stride 32) for EfficientNets
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(3, 4)
        )

        # Dynamically determine channel counts
        feature_info = self.encoder.feature_info
        c4_channels = feature_info[3]["num_chs"]
        c5_channels = feature_info[4]["num_chs"]

        self.combined_channels = c4_channels + c5_channels

        # Spatial Attention Pooling
        self.spatial_attention = SoftSpatialAttention(self.combined_channels)

    def forward(self, x):
        # x: (B, 3, H, W)

        # Extract features
        features = self.encoder(x)
        p4 = features[0]  # Stride 16
        p5 = features[1]  # Stride 32

        # Upsample P5 to match P4 spatial resolution
        p5_up = F.interpolate(
            p5, size=p4.shape[-2:], mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        # Shape: (B, C4+C5, H/16, W/16)
        fused = torch.cat([p4, p5_up], dim=1)

        # Apply Soft Spatial Attention Pooling
        # Shape: (B, C4+C5)
        embedding = self.spatial_attention(fused)

        return embedding


class AttentionHead(nn.Module):
    """
    Sequence-to-Scalar Attention Head.
    Aggregates a sequence of embeddings into a single vector using learned attention weights,
    then projects to a probability.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Attention mechanism layers
        self.attention_V = nn.Linear(input_dim, 128)
        self.attention_U = nn.Linear(128, 1)

        # Final classification layer
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # 1. Compute Attention Scores
        # u: (B, S, 128)
        u = torch.tanh(self.attention_V(x))
        # attn_logits: (B, S, 1)
        attn_logits = self.attention_U(u)
        # alpha: (B, S, 1)
        alpha = F.softmax(attn_logits, dim=1)

        # 2. Weighted Aggregation
        # context: (B, Input_Dim)
        context = (x * alpha).sum(dim=1)

        # 3. Classification
        logits = self.classifier(context)

        # Return probability
        return torch.sigmoid(logits)


class CervicalFractureNet(nn.Module):
    """
    Calibrated 2.5D Multi-Scale Network with Soft Spatial Attention.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        self.backbone = MultiScaleBackbone(Config.BACKBONE, pretrained=True)

        # Projection layer to reduce dimensionality before LSTM
        # EfficientNet-B4 P4+P5 is ~608 channels. Projecting to 512.
        self.projection = nn.Linear(self.backbone.combined_channels, 512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # 2. Sequence Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        lstm_out_dim = Config.LSTM_HIDDEN_SIZE * 2

        # 3. Positional Embeddings
        # Learnable vector added to LSTM outputs to distinguish anatomical levels
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Class-Specific Heads
        # 8 separate heads: C1-C7 + Patient Overall
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        # Input x: (Batch, Seq_Len, Channels, Height, Width)
        B, S, C, H, W = x.shape

        # --- Feature Extraction ---
        # Reshape to treat all slices as a batch: (B*S, C, H, W)
        x_flat = x.view(B * S, C, H, W)

        # CNN Backbone + Spatial Attention
        # Cite debug_lesson_1: Effective batch size B*S=192 causes OOM.
        # Cite debug_lesson_10: Chunking + Gradient Checkpointing reduces memory.
        if self.training:
            x.requires_grad_(True)
            chunk_size = 32
            features_list = []
            for i in range(0, B * S, chunk_size):
                chunk = x_flat[i : i + chunk_size]
                feat = checkpoint(self.backbone, chunk, use_reentrant=False)
                features_list.append(feat)
            features = torch.cat(features_list, dim=0)
        else:
            features = self.backbone(x_flat)  # (B*S, Combined_Channels)

        # Projection & Dropout
        features = self.projection(features)  # (B*S, 512)
        features = self.dropout(features)

        # --- Sequence Modeling ---
        # Reshape back to sequence: (B, S, 512)
        features_seq = features.view(B, S, -1)

        # Bi-LSTM
        # lstm_out: (B, S, 2*Hidden)
        lstm_out, _ = self.lstm(features_seq)

        # Inject Positional Embeddings
        # Broadcasting adds (1, S, Dim) to (B, S, Dim)
        lstm_out = lstm_out + self.pos_embed

        # --- Classification Heads ---
        # Each head attends to the sequence independently
        outputs = []
        for head in self.heads:
            # head output: (B, 1)
            outputs.append(head(lstm_out))

        # Concatenate all predictions: (B, 8)
        return torch.cat(outputs, dim=1)

    def train(self, mode=True):
        super().train(mode)
        # Cite debug_lesson_12: Freeze Batch Normalization layers in the backbone.
        # Small batch sizes with highly correlated samples (CT slices) corrupt
        # BN statistics (running_mean/var), leading to NaN during validation.
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
