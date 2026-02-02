import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Computes a weighted average of the sequence (context vector) using learnable attention weights.
    This allows specific heads to focus on specific parts of the spine (e.g., C1 vs C7).
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
        Returns:
            context: (Batch, Input_Dim)
        """
        # Compute attention scores
        scores = self.score_net(x)  # (B, S, 1)
        weights = torch.softmax(scores, dim=1)  # (B, S, 1)

        # Weighted sum of the sequence
        context = torch.sum(x * weights, dim=1)  # (B, D)
        return context


class RSNAModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # --- 1. Backbone (EfficientNet-B4) ---
        # features_only=True returns a list of feature maps.
        # out_indices=(3, 4) corresponds to P4 (stride 16) and P5 (stride 32).
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Dynamically determine feature channels from the backbone
        feature_info = self.backbone.feature_info
        p4_channels = feature_info[0]["num_chs"]
        p5_channels = feature_info[1]["num_chs"]

        # We concatenate GAP(P4) and GAP(P5)
        self.embedding_dim = p4_channels + p5_channels

        # --- 2. Sequence Modeling (Bi-LSTM) ---
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            bidirectional=Config.BIDIRECTIONAL,
            batch_first=True,
            dropout=Config.DROPOUT_RATE if Config.LSTM_LAYERS > 1 else 0.0,
        )

        self.lstm_out_dim = Config.LSTM_HIDDEN_SIZE * (2 if Config.BIDIRECTIONAL else 1)

        # --- 3. Positional Injection ---
        # Learnable position embeddings added to LSTM output to provide Z-axis awareness
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, self.lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 4. Disentangled Attention Heads ---
        # 8 Independent heads: C1, C2, C3, C4, C5, C6, C7, Patient_Overall
        self.attention_heads = nn.ModuleList(
            [AttentionHead(self.lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

        # --- 5. Classifiers ---
        # Map context vector to logit for each class independently
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.lstm_out_dim, 1) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            logits: (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Flatten Batch and Sequence dimensions to process slices as a batch of images
        x = x.view(b * s, c, h, w)

        # Extract Features (P4 and P5)
        features = self.backbone(x)
        p4 = features[0]  # (B*S, p4_c, H/16, W/16)
        p5 = features[1]  # (B*S, p5_c, H/32, W/32)

        # Global Average Pooling
        p4_gap = p4.mean(dim=(2, 3))  # (B*S, p4_c)
        p5_gap = p5.mean(dim=(2, 3))  # (B*S, p5_c)

        # Multi-Level Aggregation: Concatenate P4 and P5 features
        embeddings = torch.cat([p4_gap, p5_gap], dim=1)  # (B*S, p4_c + p5_c)

        # Restore Sequence Dimension
        embeddings = embeddings.view(b, s, -1)  # (B, S, Embed_Dim)

        # Sequence Modeling
        lstm_out, _ = self.lstm(embeddings)  # (B, S, LSTM_Out_Dim)

        # Add Positional Embeddings
        # Ensure alignment with sequence length (usually fixed to 96 by Config)
        if s <= self.pos_embed.shape[1]:
            lstm_out = lstm_out + self.pos_embed[:, :s, :]
        else:
            # Fallback for unexpected lengths (though Dataset guarantees SEQ_LEN)
            lstm_out = lstm_out + self.pos_embed

        # Attention Aggregation & Classification
        logits_list = []
        for i in range(Config.NUM_CLASSES):
            # Independent Attention Head
            context = self.attention_heads[i](lstm_out)  # (B, LSTM_Out_Dim)

            # Independent Classifier
            logit = self.classifiers[i](context)  # (B, 1)
            logits_list.append(logit)

        # Concatenate Logits: [C1, C2, ..., C7, Overall]
        logits = torch.cat(logits_list, dim=1)  # (B, 8)

        return logits
