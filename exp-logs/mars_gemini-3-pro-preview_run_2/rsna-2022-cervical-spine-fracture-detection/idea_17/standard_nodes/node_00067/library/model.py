import torch
import torch.nn as nn
import timm
from library.config import Config


class ClassSpecificAttention(nn.Module):
    """
    Computes a class-specific attention over the sequence to aggregate
    temporal features into a single study-level embedding.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # Attention mechanism: computes a scalar weight for each timestep
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        # Classifier: projects the weighted context vector to a logit
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # Calculate attention scores
        # scores shape: (Batch, Seq_Len, 1)
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)

        # Compute weighted sum (context vector)
        # context shape: (Batch, Input_Dim)
        context = torch.sum(weights * x, dim=1)

        # Compute logit
        # logit shape: (Batch, 1)
        logit = self.classifier(context)
        return logit


class Calibrated25DModel(nn.Module):
    """
    Calibrated 2.5D Multi-Level Feature Fusion Network.

    Architecture:
    1. 2.5D Input (3 slices).
    2. EfficientNet-B4 Backbone (Multi-level features P4 & P5).
    3. Global Average Pooling & Concatenation.
    4. Bidirectional LSTM.
    5. Learnable Positional Embeddings.
    6. 8 Independent Class-Specific Attention Heads.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Extract features from the last two blocks (indices 3 and 4)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.IN_CHANS,
        )

        # Enable Gradient Checkpointing to save memory
        # Cite debug_lesson_1 (Effective Batch Size) and debug_lesson_10 (Chunking vs Checkpointing)
        # This is crucial when processing sequences of images (e.g. 96 slices)
        self.backbone.set_grad_checkpointing(True)

        # Determine embedding dimensions dynamically from the backbone
        feature_channels = self.backbone.feature_info.channels()
        p4_dim = feature_channels[0]
        p5_dim = feature_channels[1]
        total_cnn_embed_dim = p4_dim + p5_dim

        # 2. Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=total_cnn_embed_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=Config.DROPOUT,
        )

        # LSTM output dimension (Bidirectional = 2 * Hidden)
        lstm_out_dim = Config.HIDDEN_DIM * 2

        # 3. Positional Embeddings
        # Learnable vector added to LSTM outputs to distinguish C1 from C7
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Heads
        # 8 Independent heads for C1-C7 and Patient Overall
        self.heads = nn.ModuleList(
            [ClassSpecificAttention(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, C, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Combine Batch and Sequence dimensions for CNN processing
        # Shape: (Batch * Seq_Len, C, H, W)
        x = x.view(b * s, c, h, w)

        # --- Feature Extraction ---
        # features is a list of tensors [P4, P5]
        features = self.backbone(x)
        p4 = features[0]  # Shape: (B*S, p4_dim, H', W')
        p5 = features[1]  # Shape: (B*S, p5_dim, H'', W'')

        # --- Multi-Level Pooling ---
        # Global Average Pooling on spatial dimensions
        p4_gap = p4.mean(dim=(2, 3))  # Shape: (B*S, p4_dim)
        p5_gap = p5.mean(dim=(2, 3))  # Shape: (B*S, p5_dim)

        # Concatenate features
        # Shape: (B*S, p4_dim + p5_dim)
        embedding = torch.cat([p4_gap, p5_gap], dim=1)

        # --- Sequence Modeling ---
        # Reshape back to sequence format
        # Shape: (Batch, Seq_Len, Embed_Dim)
        embedding = embedding.view(b, s, -1)

        # Pass through LSTM
        # lstm_out Shape: (Batch, Seq_Len, Hidden*2)
        lstm_out, _ = self.lstm(embedding)

        # Add Positional Embeddings
        # Broadcasting adds (1, Seq, Dim) to (Batch, Seq, Dim)
        lstm_out = lstm_out + self.pos_embed

        # --- Aggregation & Classification ---
        # Pass through each specific head
        logits_list = []
        for head in self.heads:
            # Each head returns (Batch, 1)
            logits_list.append(head(lstm_out))

        # Concatenate all logits
        # Shape: (Batch, 8)
        logits = torch.cat(logits_list, dim=1)

        return logits
