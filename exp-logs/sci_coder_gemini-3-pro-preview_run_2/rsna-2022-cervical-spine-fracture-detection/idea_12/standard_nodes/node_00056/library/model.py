import torch
import torch.nn as nn
import timm
from library.config import Config


class TimeDistributed(nn.Module):
    """
    Wraps a module to apply it to every timestep of a sequence independently.
    Input: (Batch, Seq, Channels, H, W)
    Output: (Batch, Seq, Features)
    """

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: (B, S, C, H, W)
        b, s, c, h, w = x.shape

        # Collapse Batch and Seq dimensions to treat slices as independent images
        x_reshaped = x.view(b * s, c, h, w)

        # Pass through the module (Backbone)
        y = self.module(x_reshaped)

        # y shape: (B*S, Feature_Dim)
        feature_dim = y.shape[1]

        # Reshape back to (B, S, Feature_Dim)
        y_reshaped = y.view(b, s, feature_dim)
        return y_reshaped


class MultiLevelEfficientNet(nn.Module):
    """
    EfficientNet backbone that extracts and aggregates features from multiple levels (P4 and P5).
    Uses Global Average Pooling on each level and concatenates them to preserve both
    high-level semantics and fine-grained texture details.
    """

    def __init__(self, model_name, pretrained=True):
        super(MultiLevelEfficientNet, self).__init__()

        # Load model with features_only=True to access intermediate layers
        # out_indices=(3, 4) targets the last two blocks:
        # Index 3: Stride 16 (P4)
        # Index 4: Stride 32 (P5)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Cite debug_lesson_1: Enable gradient checkpointing to reduce memory of effective batch
        # Cite debug_lesson_5: Use set_grad_checkpointing API instead of constructor arg
        self.backbone.set_grad_checkpointing(True)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Dynamically determine output dimension
        self.out_features = self._get_out_features()

    def _get_out_features(self):
        """Helper to calculate total feature dimension after concatenation."""
        with torch.no_grad():
            # Create dummy input: (Batch, C, H, W)
            # Size 256 is arbitrary, just need to pass through layers
            dummy = torch.randn(1, Config.IN_CHANNELS, 256, 256)
            features = self.backbone(dummy)
            # features is a list of tensors [P4, P5]
            total_dim = 0
            for f in features:
                total_dim += f.shape[1]
            return total_dim

    def train(self, mode=True):
        """
        Override train mode to keep Batch Normalization layers in eval mode.
        Cite debug_lesson_12: Correlated samples in sequence cause BN instability.
        """
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        # x: (N, C, H, W)
        features = self.backbone(x)

        # features[0] is P4, features[1] is P5
        p4 = self.pool(features[0]).flatten(1)
        p5 = self.pool(features[1]).flatten(1)

        # Concatenate features: (N, Dim_P4 + Dim_P5)
        return torch.cat([p4, p5], dim=1)


class AttentionHead(nn.Module):
    """
    Class-Specific Attention Head.
    Aggregates the sequence embedding into a single vector using learned attention weights,
    then projects to a classification logit. This allows different heads to focus on
    different parts of the spine (e.g., C1 vs C7).
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionHead, self).__init__()

        # Attention mechanism: u_t = tanh(W x_t + b)
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # Final classifier
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)

        # Calculate attention scores
        # scores: (Batch, Seq_Len, 1)
        scores = self.attention_layer(x)

        # Normalize to weights
        weights = torch.softmax(scores, dim=1)

        # Weighted sum (Context vector)
        # context: (Batch, Input_Dim)
        context = torch.sum(x * weights, dim=1)

        # Classification logit
        # output: (Batch, 1)
        output = self.classifier(context)

        return output


class CalibratedSequenceModel(nn.Module):
    """
    Calibrated 2.5D Multi-Level Sequence Network (Idea 12).

    Architecture:
    1. 2.5D Input (Stack of 3 slices)
    2. Multi-Level EfficientNet Backbone (P4+P5 GAP)
    3. Bidirectional LSTM
    4. Positional Embedding Injection
    5. 8 Independent Attention Heads (C1-C7, Patient Overall)
    """

    def __init__(self):
        super(CalibratedSequenceModel, self).__init__()

        # --- 1. Feature Extractor ---
        self.feature_extractor = TimeDistributed(
            MultiLevelEfficientNet(Config.BACKBONE, pretrained=True)
        )
        cnn_out_dim = self.feature_extractor.module.out_features

        # --- 2. Sequence Modeling ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )

        # --- 3. Positional Injection ---
        # Learnable vector added to LSTM outputs to distinguish anatomical level (top vs bottom)
        # Shape: (1, Seq_Len, LSTM_Dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, lstm_out_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # --- 4. Disentangled Heads ---
        # Create one head for each class (C1-C7 + Patient Overall)
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels, H, W)

        # 1. Extract Features
        # Output: (Batch, Seq_Len, CNN_Dim)
        x = self.feature_extractor(x)

        # 2. LSTM
        # Output: (Batch, Seq_Len, LSTM_Dim)
        x, _ = self.lstm(x)

        # 3. Add Positional Embedding
        # Broadcasts across batch dimension
        # Ensure sequence length matches (Config.SEQ_LEN ensures this in DataLoader)
        if x.size(1) == self.pos_embedding.size(1):
            x = x + self.pos_embedding
        else:
            # Fallback for varying lengths if config changes
            x = x + self.pos_embedding[:, : x.size(1), :]

        # 4. Apply Attention Heads
        logits_list = []
        for head in self.heads:
            # Each head outputs (Batch, 1)
            logits_list.append(head(x))

        # Concatenate to (Batch, 8)
        logits = torch.cat(logits_list, dim=1)

        # 5. Output Logits
        # Cite debug_lesson_13: Return raw logits for BCEWithLogitsLoss compatibility
        return logits
