import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Computes a weighted average of the sequence (context vector),
    projects it to a lower-dimensional embedding, and generates a classification logit.

    This component corresponds to the 'Level 1 (Disentangled Subtypes)' in the architecture.
    """

    def __init__(self, input_dim, feature_dim, dropout=0.0):
        super(AttentionHead, self).__init__()

        # Attention mechanism
        # u = tanh(W * h + b)
        # alpha = softmax(v^T * u)
        self.attention_proj = nn.Linear(input_dim, input_dim)
        self.attention_v = nn.Linear(input_dim, 1, bias=False)

        # Projection to specific feature vector E_Cx
        # This embedding is used for both the specific class prediction and the global fusion
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier for the specific vertebrae
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input sequence of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            logit: Classification logit (Batch, 1)
            embedding: Projected feature vector (Batch, Feature_Dim)
        """
        # 1. Compute Attention Weights
        # u: (B, S, D)
        u = torch.tanh(self.attention_proj(x))
        # scores: (B, S, 1)
        scores = self.attention_v(u)
        # weights: (B, S, 1)
        weights = torch.softmax(scores, dim=1)

        # 2. Compute Context Vector
        # context: (B, D) = sum(weights * x)
        context = torch.sum(weights * x, dim=1)

        # 3. Project to Embedding E_Cx
        embedding = self.feature_proj(context)

        # 4. Classify
        logit = self.classifier(embedding)

        return logit, embedding


class CervicalSpineModel(nn.Module):
    """
    Calibrated 2.5D Multi-Level Network with Hierarchical Embedding Aggregation.

    Structure:
    1. 2.5D Input (3 slices) -> EfficientNet-B4 (P4+P5 features)
    2. Bi-LSTM Sequence Modeling + Positional Embeddings
    3. 7 Independent Attention Heads (C1-C7)
    4. Fusion Head (Patient Overall) aggregating C1-C7 embeddings
    """

    def __init__(self):
        super(CervicalSpineModel, self).__init__()

        # 1. Backbone (Time-Distributed EfficientNet-B4)
        # features_only=True allows extraction of P4 and P5
        self.backbone = timm.create_model(
            Config.backbone,
            pretrained=Config.pretrained,
            features_only=True,
            out_indices=(3, 4),  # P4 (stride 16) and P5 (stride 32)
            in_chans=Config.in_chans,
        )

        # Get channel dimensions for P4 and P5
        feature_info = self.backbone.feature_info.channels()
        p4_channels = feature_info[0]
        p5_channels = feature_info[1]

        # Global Average Pooling is applied to P4 and P5 independently, then concatenated
        cnn_out_dim = p4_channels + p5_channels

        # 2. Sequence Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=Config.lstm_hidden_size,
            num_layers=Config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=Config.dropout if Config.lstm_layers > 1 else 0,
        )

        lstm_out_dim = Config.lstm_hidden_size * 2

        # Positional Embeddings
        # Added to LSTM output to provide explicit spatial references for vertebrae identification
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.seq_len, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. Hierarchical Heads
        # Level 1: 7 Independent Attention Heads for C1-C7
        self.c_heads = nn.ModuleList(
            [
                AttentionHead(lstm_out_dim, Config.feature_dim, Config.dropout)
                for _ in range(7)
            ]
        )

        # Level 2: Fusion Head for Patient Overall
        # Input is concatenation of 7 embeddings (Embedding-Space Aggregation)
        fusion_input_dim = Config.feature_dim * 7
        self.patient_head = nn.Sequential(
            nn.Dropout(Config.dropout), nn.Linear(fusion_input_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            logits: Tensor of shape (Batch, 8) [C1...C7, Overall]
        """
        b, s, c, h, w = x.shape

        # --- 1. Backbone Feature Extraction ---
        # Reshape for CNN: (B*S, C, H, W)
        x = x.view(b * s, c, h, w)

        # Extract features (P4, P5)
        features = self.backbone(x)
        p4 = features[0]  # (B*S, C_p4, H/16, W/16)
        p5 = features[1]  # (B*S, C_p5, H/32, W/32)

        # Global Average Pooling
        p4_gap = torch.mean(p4, dim=(2, 3))  # (B*S, C_p4)
        p5_gap = torch.mean(p5, dim=(2, 3))  # (B*S, C_p5)

        # Concatenate multi-level features
        cnn_out = torch.cat([p4_gap, p5_gap], dim=1)  # (B*S, C_p4 + C_p5)

        # --- 2. Sequence Modeling ---
        # Reshape for LSTM: (B, S, Feature_Dim)
        lstm_in = cnn_out.view(b, s, -1)

        # LSTM Forward
        # lstm_out: (B, S, Hidden*2)
        lstm_out, _ = self.lstm(lstm_in)

        # Add Positional Embeddings
        # Broadcasts along batch dimension
        lstm_out = lstm_out + self.pos_embed

        # --- 3. Hierarchical Prediction ---
        c_logits = []
        c_embeddings = []

        # Iterate over C1-C7 heads to get specific predictions and embeddings
        for head in self.c_heads:
            logit, embed = head(lstm_out)
            c_logits.append(logit)
            c_embeddings.append(embed)

        # Stack C1-C7 logits: (B, 7)
        c_logits_stacked = torch.cat(c_logits, dim=1)

        # Fusion for Patient Overall
        # Concatenate embeddings: (B, 7 * Feature_Dim)
        fusion_in = torch.cat(c_embeddings, dim=1)
        patient_logit = self.patient_head(fusion_in)  # (B, 1)

        # Final Output: (B, 8) -> [C1, ..., C7, Patient_Overall]
        # Order matches Config.target_cols
        final_logits = torch.cat([c_logits_stacked, patient_logit], dim=1)

        return final_logits
