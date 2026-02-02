import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Class-Specific Attention Head.
    Computes independent attention weights over the sequence to aggregate features
    and produce a single logit for a specific class (e.g., C1 or Patient Overall).
    """

    def __init__(self, input_dim):
        super().__init__()
        # Attention scoring mechanism: Maps input to a scalar score per time step
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        # Classifier: Maps the context vector to a logit
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT), nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # 1. Calculate Attention Scores
        # scores shape: (Batch, Seq_Len, 1)
        scores = self.attention(x)

        # 2. Normalize to Attention Weights
        weights = torch.softmax(scores, dim=1)

        # 3. Compute Context Vector (Weighted Sum)
        # context shape: (Batch, Input_Dim)
        context = torch.sum(weights * x, dim=1)

        # 4. Classification
        # logit shape: (Batch, 1)
        logit = self.classifier(context)

        return logit


class CervicalFractureModel(nn.Module):
    """
    Calibrated 2.5D Multi-Level Sequence Network.

    Architecture components:
    1. Backbone: EfficientNet-B4 (extracting P4 and P5 features).
    2. Aggregation: Multi-Level Global Average Pooling.
    3. Sequence: Bi-LSTM with Positional Injection.
    4. Heads: 8 Independent Attention Heads.
    """

    def __init__(self):
        super().__init__()

        # --- 1. Backbone ---
        # Load EfficientNet-B4, extracting features from P4 (index 3) and P5 (index 4)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(3, 4),
        )

        # Enable Gradient Checkpointing to save memory (Cite Debug Lesson 11)
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        # Dynamically determine feature dimensions from the backbone
        # feature_info.channels() returns a list of channels for the selected out_indices
        feature_channels = self.backbone.feature_info.channels()
        p4_channels = feature_channels[0]
        p5_channels = feature_channels[1]
        total_feature_dim = p4_channels + p5_channels

        # --- 2. Sequence Encoder ---
        self.lstm = nn.LSTM(
            input_size=total_feature_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        # Bidirectional LSTM outputs 2 * hidden_size
        lstm_out_dim = Config.LSTM_HIDDEN_SIZE * 2

        # --- 3. Positional Embeddings ---
        # Learnable vector added to LSTM outputs to provide spatial context
        # Shape: (1, Seq_Len, Hidden*2)
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 4. Heads ---
        # Create 8 independent heads (C1-C7 + Patient Overall)
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

    def train(self, mode=True):
        """
        Override train mode to freeze Batch Normalization layers.
        Since Batch Size is 1, the slices are highly correlated (same patient),
        which corrupts BN statistics (Cite Debug Lesson 12).
        """
        super().train(mode)
        if mode:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, Height, Width)
               Channels is 3 (2.5D stack).
        Returns:
            logits: Output tensor of shape (Batch, Num_Classes)
        """
        b, seq, c, h, w = x.shape

        # 1. Fold Batch and Sequence dimensions for CNN processing
        # Shape: (B * Seq, C, H, W)
        x = x.view(b * seq, c, h, w)

        # 2. Backbone Forward Pass
        # Returns a list of feature maps [P4, P5]
        features = self.backbone(x)
        p4 = features[0]  # Shape: (B*S, C4, H4, W4)
        p5 = features[1]  # Shape: (B*S, C5, H5, W5)

        # 3. Multi-Level Global Average Pooling
        p4_vec = p4.mean(dim=(2, 3))  # Shape: (B*S, C4)
        p5_vec = p5.mean(dim=(2, 3))  # Shape: (B*S, C5)

        # Concatenate features
        embedding = torch.cat([p4_vec, p5_vec], dim=1)  # Shape: (B*S, C4+C5)

        # 4. Reshape for Sequence Modeling
        # Shape: (B, Seq, Feature_Dim)
        embedding = embedding.view(b, seq, -1)

        # 5. LSTM Processing
        # lstm_out shape: (B, Seq, Hidden*2)
        lstm_out, _ = self.lstm(embedding)

        # 6. Inject Positional Embeddings
        # Add learnable position vector (broadcasting over batch)
        lstm_out = lstm_out + self.pos_embed

        # 7. Independent Attention Heads
        logits_list = []
        for head in self.heads:
            logits_list.append(head(lstm_out))

        # Concatenate logits from all heads
        # Final Shape: (B, 8)
        logits = torch.cat(logits_list, dim=1)

        return logits
