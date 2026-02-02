import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CervicalFractureNet(nn.Module):
    """
    2.5D Multi-Head Attention Network for Cervical Spine Fracture Detection.

    Architecture:
    1. Time-Distributed CNN Backbone (EfficientNet-B3)
    2. Bidirectional LSTM
    3. Class-Specific Attention Heads (C1-C7)
    4. Fusion Head for Patient Overall
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Backbone (Time-Distributed)
        # =====================================================================
        # Load pre-trained EfficientNet-B3
        # num_classes=0 removes the classifier, returning the pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.cnn_feature_dim = self.backbone.num_features

        # =====================================================================
        # 2. Sequence Modeling (LSTM)
        # =====================================================================
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        # Bidirectional output dimension
        self.lstm_out_dim = Config.LSTM_HIDDEN_SIZE * 2

        # =====================================================================
        # 3. Class-Specific Attention Mechanism
        # =====================================================================
        # We have 7 specific targets: C1, C2, C3, C4, C5, C6, C7.
        # Each gets its own attention head to extract relevant features from the sequence.
        self.num_vertebrae = 7

        # Learnable Query Vectors: Shape (7, LSTM_Out_Dim)
        # These act as "prototypes" for what each vertebra's features look like.
        self.attention_queries = nn.Parameter(
            torch.randn(self.num_vertebrae, self.lstm_out_dim)
        )

        # Scaling factor for dot-product attention
        self.scale = self.lstm_out_dim**-0.5

        # =====================================================================
        # 4. Prediction Heads
        # =====================================================================

        # A. Vertebrae-Specific Classifiers (C1-C7)
        # We use a ModuleList of Linear layers, one for each vertebra.
        # They take the vertebra-specific context vector and output a logit.
        self.vertebrae_classifiers = nn.ModuleList(
            [nn.Linear(self.lstm_out_dim, 1) for _ in range(self.num_vertebrae)]
        )

        # B. Patient Overall Classifier
        # Aggregates the 7 context vectors to make a global prediction.
        # Input dim: 7 * LSTM_Out_Dim
        self.overall_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.num_vertebrae * self.lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, 8).
                          Columns: [C1, C2, C3, C4, C5, C6, C7, Patient_Overall]
        """
        b, seq_len, c, h, w = x.shape

        # ---------------------------------------------------------------------
        # 1. Extract Features (Time-Distributed CNN)
        # ---------------------------------------------------------------------
        # Merge Batch and Sequence dimensions: (B*S, C, H, W)
        x_reshaped = x.view(b * seq_len, c, h, w)

        # Pass through backbone
        cnn_features = self.backbone(x_reshaped)  # (B*S, CNN_Dim)

        # Reshape back to sequence: (B, S, CNN_Dim)
        cnn_features = cnn_features.view(b, seq_len, -1)

        # ---------------------------------------------------------------------
        # 2. Sequence Modeling (LSTM)
        # ---------------------------------------------------------------------
        # lstm_out: (B, S, LSTM_Dim * 2)
        lstm_out, _ = self.lstm(cnn_features)

        # ---------------------------------------------------------------------
        # 3. Class-Specific Attention
        # ---------------------------------------------------------------------
        # We calculate attention weights for each of the 7 vertebrae.
        # Query: (7, Dim)
        # Key (Sequence): (B, S, Dim)

        # Calculate Scores: (B, S, Dim) @ (Dim, 7) -> (B, S, 7)
        # We transpose the queries to (Dim, 7)
        attn_scores = torch.matmul(lstm_out, self.attention_queries.t())

        # Apply scaling
        attn_scores = attn_scores * self.scale

        # Softmax over the Sequence dimension (dim=1)
        # Result: (B, S, 7) - How much each time step contributes to each vertebra
        attn_weights = F.softmax(attn_scores, dim=1)

        # Calculate Context Vectors
        # We want (B, 7, Dim).
        # Permute weights to (B, 7, S)
        # Multiply by LSTM output (B, S, Dim)
        # (B, 7, S) @ (B, S, Dim) -> (B, 7, Dim)
        context_vectors = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out)

        # ---------------------------------------------------------------------
        # 4. Predictions
        # ---------------------------------------------------------------------
        logits_list = []

        # A. C1-C7 Predictions
        for i in range(self.num_vertebrae):
            # Extract context for vertebra i: (B, Dim)
            ctx = context_vectors[:, i, :]

            # Pass through specific classifier
            logit = self.vertebrae_classifiers[i](ctx)  # (B, 1)
            logits_list.append(logit)

        # B. Patient Overall Prediction
        # Flatten all context vectors: (B, 7 * Dim)
        overall_input = context_vectors.view(b, -1)
        overall_logit = self.overall_classifier(overall_input)  # (B, 1)
        logits_list.append(overall_logit)

        # Concatenate all logits: (B, 8)
        # Order matches data loader: C1, C2, C3, C4, C5, C6, C7, Overall
        final_logits = torch.cat(logits_list, dim=1)

        return final_logits
