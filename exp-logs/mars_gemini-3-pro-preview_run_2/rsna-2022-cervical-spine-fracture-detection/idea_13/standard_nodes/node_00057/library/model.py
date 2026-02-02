import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EfficientNetFeatureExtractor(nn.Module):
    """
    Extracts features from EfficientNet-B4.
    Uses P4 and P5 blocks, applies GAP, and concatenates them.
    """

    def __init__(self, model_name=Config.BACKBONE, pretrained=True):
        super().__init__()
        # Load model with features_only=True to get intermediate feature maps
        # out_indices=(3, 4) corresponds to P4 and P5 for EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Determine output feature dimension dynamically
        # We run a dummy pass to get channel counts
        with torch.no_grad():
            dummy = torch.zeros(1, Config.IN_CHANNELS, 256, 256)
            feats = self.backbone(dummy)
            # feats[0] is P4, feats[1] is P5
            self.p4_dim = feats[0].shape[1]
            self.p5_dim = feats[1].shape[1]

        self.out_dim = self.p4_dim + self.p5_dim

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (Batch * Seq, C, H, W)
        feats = self.backbone(x)
        p4 = feats[0]  # (B*S, p4_dim, H/16, W/16)
        p5 = feats[1]  # (B*S, p5_dim, H/32, W/32)

        # Apply GAP and flatten
        p4_vec = self.gap(p4).flatten(1)
        p5_vec = self.gap(p5).flatten(1)

        # Concatenate
        return torch.cat([p4_vec, p5_vec], dim=1)


class AttentionHead(nn.Module):
    """
    Computes attention-weighted average of the sequence and classification logit.
    Returns both logit and attention weights for supervision.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # Compute attention scores
        # attn_scores: (Batch, Seq, 1)
        attn_scores = self.attention(x)

        # Normalize to weights
        # weights: (Batch, Seq, 1)
        weights = F.softmax(attn_scores, dim=1)

        # Weighted sum (Context Vector)
        # context: (Batch, Input_Dim)
        context = torch.sum(x * weights, dim=1)

        # Classification
        logit = self.classifier(context)

        return logit, weights.squeeze(-1)


class CervicalFractureNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone
        self.feature_extractor = EfficientNetFeatureExtractor()
        cnn_out_dim = self.feature_extractor.out_dim

        # 2. Sequence Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
        )

        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )

        # 3. Positional Embeddings
        # Learnable vector added to LSTM outputs to distinguish anatomical height
        self.pos_embedding = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 4. Heads
        # We need 8 heads: C1..C7 and patient_overall
        # We store them in a ModuleList
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (Batch, Seq, C, H, W)

        Returns:
            dict: {
                "logits": (Batch, 8),
                "attn_weights": (Batch, 8, Seq)
            }
        """
        b, s, c, h, w = x.shape

        # --- Feature Extraction ---
        # Merge Batch and Seq dimensions for CNN
        x = x.view(b * s, c, h, w)

        # Extract features: (B*S, Feature_Dim)
        features = self.feature_extractor(x)

        # Reshape back to sequence: (B, S, Feature_Dim)
        features = features.view(b, s, -1)

        # --- Sequence Modeling ---
        # LSTM output: (B, S, Hidden*2)
        self.lstm.flatten_parameters()
        seq_features, _ = self.lstm(features)

        # Add Positional Embeddings
        seq_features = seq_features + self.pos_embedding

        # --- Heads ---
        logits_list = []
        attn_weights_list = []

        for head in self.heads:
            logit, weights = head(seq_features)
            logits_list.append(logit)
            attn_weights_list.append(weights)

        # Stack results
        # logits: (Batch, 8)
        logits = torch.cat(logits_list, dim=1)

        # attn_weights: (Batch, 8, Seq)
        attn_weights = torch.stack(attn_weights_list, dim=1)

        return {"logits": logits, "attn_weights": attn_weights}
