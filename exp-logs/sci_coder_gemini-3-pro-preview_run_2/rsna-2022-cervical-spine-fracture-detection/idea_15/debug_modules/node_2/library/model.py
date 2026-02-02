import torch
import torch.nn as nn
import timm
from library.config import Config


class CalibratedSequenceNetwork(nn.Module):
    """
    Calibrated 2.5D Multi-Scale Sequence Network.

    This model implements a sequence-based approach for fracture detection, featuring:
    1. 2.5D Multi-Scale Backbone: Extracts P4 and P5 features from EfficientNet-B4.
    2. Multi-Level Adapter: Aggregates multi-scale features via GAP and concatenation.
    3. Sequence Encoder: Bi-LSTM with learnable positional injections.
    4. Disentangled Attention: Independent attention heads for each vertebrae/target.
    5. Independent Classifiers: Separate linear projections for each target.
    """

    def __init__(self):
        super(CalibratedSequenceNetwork, self).__init__()

        # 1. Backbone
        # We use features_only=True to extract intermediate feature maps.
        # out_indices=(3, 4) corresponds to the stride 16 (P4) and stride 32 (P5) blocks.
        # This ensures we capture both fine fracture details and global context.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimensions dynamically based on the backbone architecture
        # feature_info.channels() returns the channel counts for the selected out_indices.
        feature_channels = self.backbone.feature_info.channels()
        p4_dim = feature_channels[0]
        p5_dim = feature_channels[1]

        # The adapter concatenates GAP pooled P4 and P5 vectors
        self.adapter_dim = p4_dim + p5_dim

        # 2. Sequence Encoder (Bi-LSTM)
        # Models the anatomical continuity along the Z-axis.
        self.lstm = nn.LSTM(
            input_size=self.adapter_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        self.lstm_dim = Config.LSTM_HIDDEN_SIZE * 2

        # 3. Positional Embeddings
        # Learnable vector added to LSTM outputs to provide explicit spatial reference.
        # Shape: (1, Seq_Len, LSTM_Dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, self.lstm_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 4. Disentangled Attention Aggregation
        # We project LSTM outputs to 'Num_Classes' scores.
        # Each class has its own attention distribution over the sequence.
        # This allows C1 to focus on the top slices and C7 on the bottom slices independently.
        self.attention_proj = nn.Linear(self.lstm_dim, Config.NUM_CLASSES)

        # 5. Classifiers
        # Independent linear classifier for each class.
        # Input: Aggregated feature vector (LSTM_Dim). Output: Logit (1).
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.lstm_dim, 1) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Channels, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        b, s, c, h, w = x.shape

        # --- Feature Extraction ---
        # Flatten batch and sequence dimensions for efficient CNN processing
        # Shape: (Batch * Seq_Len, Channels, H, W)
        x = x.view(b * s, c, h, w)

        # Extract features (P4, P5)
        features = self.backbone(x)
        p4 = features[0]  # (B*S, P4_Dim, H/16, W/16)
        p5 = features[1]  # (B*S, P5_Dim, H/32, W/32)

        # Global Average Pooling
        p4 = p4.mean(dim=(2, 3))  # (B*S, P4_Dim)
        p5 = p5.mean(dim=(2, 3))  # (B*S, P5_Dim)

        # Multi-Level Aggregation: Concatenate P4 and P5
        embeddings = torch.cat([p4, p5], dim=1)  # (B*S, Adapter_Dim)

        # Reshape back to sequence format for LSTM
        embeddings = embeddings.view(b, s, -1)  # (B, S, Adapter_Dim)

        # --- Sequence Modeling ---
        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(embeddings)  # (B, S, LSTM_Dim)

        # --- Positional Injection ---
        # Add learnable position embeddings to the sequence features
        # pos_embed broadcasts over the batch dimension
        lstm_out = lstm_out + self.pos_embed
        lstm_out = self.dropout(lstm_out)

        # --- Disentangled Attention Aggregation ---
        # Compute raw attention scores for each class
        # attn_logits: (B, S, Num_Classes)
        attn_logits = self.attention_proj(lstm_out)

        # Normalize scores over the sequence dimension (dim=1) to get probabilities
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, S, Num_Classes)

        # Weighted Sum Aggregation
        # For each class c, compute the weighted sum of LSTM outputs:
        # Feature_c = Sum_over_t( Weight_t,c * LSTM_Output_t )
        # We use einsum for clarity and efficiency:
        # b=batch, s=sequence, d=dimension, c=class
        weighted_features = torch.einsum("bsd,bsc->bcd", lstm_out, attn_weights)

        # --- Classification ---
        logits_list = []
        for k in range(Config.NUM_CLASSES):
            # Extract the aggregated feature vector specific to class k
            k_feat = weighted_features[:, k, :]  # (B, LSTM_Dim)

            # Pass through the class-specific linear layer
            k_logit = self.classifiers[k](k_feat)  # (B, 1)
            logits_list.append(k_logit)

        # Concatenate all logits to form the final output
        logits = torch.cat(logits_list, dim=1)  # (B, Num_Classes)

        return logits
