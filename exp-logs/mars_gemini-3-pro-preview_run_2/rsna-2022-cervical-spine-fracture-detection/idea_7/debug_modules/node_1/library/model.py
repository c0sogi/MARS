import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Class-Specific Attention Head.
    Computes a weighted sum of the sequence embeddings (context vector)
    and projects it to a single class logit.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionHead, self).__init__()
        # Attention scoring network: Maps input -> scalar score
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        # Final classification layer applied to the context vector
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # 1. Calculate Attention Scores
        # scores: (Batch, Seq_Len, 1)
        scores = self.attention(x)

        # 2. Normalize to Weights via Softmax
        # weights: (Batch, Seq_Len, 1)
        weights = torch.softmax(scores, dim=1)

        # 3. Compute Context Vector (Weighted Sum)
        # context: (Batch, Input_Dim)
        context = torch.sum(x * weights, dim=1)

        # 4. Classification
        # logits: (Batch, 1)
        logits = self.classifier(context)

        return logits


class CervicalSpineModel(nn.Module):
    """
    2.5D Multi-Scale Sequence Network for Fracture Detection.

    Architecture:
    1. EfficientNet-B4 Backbone (Multi-Scale: P4 + P5 features)
    2. Bi-LSTM Sequence Modeler
    3. Learnable Positional Embeddings
    4. 8x Class-Specific Attention Heads (Study Prediction)
    5. 1x Dense Linear Head (Auxiliary Slice Prediction)
    """

    def __init__(self):
        super(CervicalSpineModel, self).__init__()

        # --- 1. Backbone (EfficientNet-B4) ---
        # features_only=True allows accessing intermediate feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve channel counts for P4 (index -2) and P5 (index -1)
        # feature_info.channels() returns a list of channel depths for each extracted feature map
        feature_channels = self.backbone.feature_info.channels()
        p4_channels = feature_channels[-2]
        p5_channels = feature_channels[-1]

        # Total dimension after concatenating GAP(P4) and GAP(P5)
        self.embedding_dim = p4_channels + p5_channels

        # --- 2. Sequence Modeling (Bi-LSTM) ---
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROP_RATE,
        )

        # LSTM output dimension (Bidirectional = x2)
        lstm_out_dim = Config.HIDDEN_DIM * 2

        # --- 3. Positional Embeddings ---
        # Learnable vector added to LSTM outputs to encode Z-axis position
        # Shape: (1, SEQ_LEN, LSTM_Out_Dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 4. Heads ---
        # Primary Heads: 8 separate attention mechanisms for C1-C7 and Overall
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

        # Auxiliary Head: Dense supervision for slice-level fracture probability
        self.aux_head = nn.Linear(lstm_out_dim, 1)

        # Regularization
        self.dropout = nn.Dropout(Config.DROP_RATE)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Seq_Len, C, H, W)

        Returns:
            study_logits (torch.Tensor): Shape (Batch, 8)
            slice_logits (torch.Tensor): Shape (Batch, Seq_Len)
        """
        b, s, c, h, w = x.shape

        # --- 1. Feature Extraction ---
        # Reshape to (Batch * Seq_Len, C, H, W) for 2D CNN processing
        x = x.view(b * s, c, h, w)

        # Forward pass through backbone
        # features is a list of tensors corresponding to selected stages
        features = self.backbone(x)

        # Extract P4 and P5
        p4 = features[-2]  # Shape: (B*S, C4, H/16, W/16)
        p5 = features[-1]  # Shape: (B*S, C5, H/32, W/32)

        # Global Average Pooling
        p4_pooled = p4.mean(dim=(-2, -1))  # (B*S, C4)
        p5_pooled = p5.mean(dim=(-2, -1))  # (B*S, C5)

        # Multi-Scale Fusion: Concatenate features
        embeddings = torch.cat([p4_pooled, p5_pooled], dim=1)  # (B*S, C4+C5)

        # Reshape back to sequence format: (Batch, Seq_Len, Emb_Dim)
        embeddings = embeddings.view(b, s, -1)

        # --- 2. Sequence Modeling ---
        # LSTM Output: (Batch, Seq_Len, Hidden*2)
        lstm_out, _ = self.lstm(embeddings)

        # Add Positional Embeddings
        # Safe slicing in case input sequence length < Config.SEQ_LEN
        if s <= self.pos_embed.shape[1]:
            lstm_out = lstm_out + self.pos_embed[:, :s, :]
        else:
            # If input is longer than config (unlikely), repeat or fail.
            # Here we just take what we have, but usually s == SEQ_LEN.
            lstm_out = lstm_out + self.pos_embed

        lstm_out = self.dropout(lstm_out)

        # --- 3. Output Heads ---

        # Auxiliary Head (Slice-Level)
        # Output: (Batch, Seq_Len, 1) -> Squeeze to (Batch, Seq_Len)
        slice_logits = self.aux_head(lstm_out).squeeze(-1)

        # Primary Heads (Study-Level)
        study_logits_list = []
        for head in self.heads:
            # Each head returns (Batch, 1)
            study_logits_list.append(head(lstm_out))

        # Concatenate to (Batch, 8)
        study_logits = torch.cat(study_logits_list, dim=1)

        return study_logits, slice_logits
