import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionHead(nn.Module):
    """
    Class-specific attention head.
    Computes a weighted average of the sequence outputs and then classifies.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionHead, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
        Returns:
            logit: (Batch, 1)
        """
        # Calculate attention scores
        # x shape: (B, S, D) -> attn shape: (B, S, 1)
        attn_scores = self.attention(x)

        # Normalize scores to weights across the sequence dimension (dim=1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Compute context vector (weighted sum)
        # (B, S, 1) * (B, S, D) -> (B, S, D) -> sum(dim=1) -> (B, D)
        context = torch.sum(attn_weights * x, dim=1)

        # Classification
        logit = self.classifier(context)  # (B, 1)

        return logit


class CervicalFractureNet(nn.Module):
    """
    2.5D Anatomically-Guided Attention Network.

    Architecture:
    1. 2.5D Backbone (EfficientNet-B4) -> Extracts features from slice triplets.
    2. Auxiliary Head -> Predicts anatomical level (C1-C7) for supervision and feature injection.
    3. Feature Fusion -> Concatenates visual features with anatomical probabilities.
    4. Bi-LSTM -> Models sequential dependencies along the Z-axis.
    5. Multi-Head Attention -> 8 Independent heads for each target class.
    """

    def __init__(self, config=None):
        super(CervicalFractureNet, self).__init__()

        if config is None:
            self.config = Config()
        else:
            self.config = config

        # --- 1. Backbone ---
        # Using timm to load EfficientNet
        # num_classes=0 returns the pooled feature vector (no classifier)
        # in_chans=3 corresponds to the 2.5D stacking (z-1, z, z+1)
        self.backbone = timm.create_model(
            self.config.BACKBONE_NAME,
            pretrained=self.config.BACKBONE_PRETRAINED,
            num_classes=0,
            in_chans=self.config.IN_CHANNELS,
            global_pool="avg",
        )

        # Enable gradient checkpointing to reduce memory usage. Cite debug_lesson_1
        # Check for method/attribute to avoid TypeError. Cite debug_lesson_5
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        elif hasattr(self.backbone, "grad_checkpointing"):
            self.backbone.grad_checkpointing = True

        self.feature_dim = self.backbone.num_features

        # --- 2. Auxiliary Head ---
        # Predicts anatomical location (C1-C7, Background) per slice
        self.aux_head = nn.Linear(self.feature_dim, self.config.AUX_NUM_CLASSES)

        # --- 3. Sequence Modeling (LSTM) ---
        # Input: CNN Features + Anatomical Probabilities (Fusion)
        lstm_input_size = self.feature_dim + self.config.AUX_NUM_CLASSES

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=self.config.BIDIRECTIONAL,
            dropout=self.config.DROP_RATE if self.config.LSTM_LAYERS > 1 else 0,
        )

        lstm_output_dim = self.config.LSTM_HIDDEN_SIZE * (
            2 if self.config.BIDIRECTIONAL else 1
        )

        # --- 4. Prediction Heads ---
        # 8 Independent Attention Heads (C1...C7, Patient_Overall)
        # Each head learns to attend to different parts of the sequence
        self.heads = nn.ModuleList(
            [AttentionHead(lstm_output_dim) for _ in range(self.config.NUM_CLASSES)]
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            dict:
                - 'fracture_logits': (Batch, Num_Classes)
                - 'aux_logits': (Batch, Seq_Len, Aux_Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Flatten batch and sequence dimensions for CNN processing
        # We treat every slice in every study as an independent image for the backbone
        x_flat = x.view(b * s, c, h, w)

        # 1. Extract Features
        features = self.backbone(x_flat)  # (B*S, Feature_Dim)

        # 2. Auxiliary Task
        aux_logits_flat = self.aux_head(features)  # (B*S, Aux_Classes)

        # Compute probabilities to inject into the main stream
        aux_probs_flat = torch.softmax(aux_logits_flat, dim=1)

        # 3. Feature Fusion
        # Inject anatomical knowledge into the feature stream
        # This tells the LSTM explicitly "This slice looks like C2"
        fused_features = torch.cat(
            [features, aux_probs_flat], dim=1
        )  # (B*S, Feature_Dim + Aux_Classes)

        # Reshape back to sequence format for LSTM
        # (B, S, Feature_Dim + Aux_Classes)
        lstm_input = fused_features.view(b, s, -1)

        # 4. Sequence Modeling
        self.lstm.flatten_parameters()  # Optimization for RNNs
        lstm_out, _ = self.lstm(lstm_input)  # (B, S, LSTM_Out_Dim)

        # 5. Class-Specific Attention & Prediction
        fracture_logits_list = []
        for head in self.heads:
            # Each head outputs (B, 1)
            fracture_logits_list.append(head(lstm_out))

        # Concatenate to (B, Num_Classes)
        fracture_logits = torch.cat(fracture_logits_list, dim=1)

        # Reshape aux logits for loss calculation: (B*S, Aux_C) -> (B, S, Aux_C)
        aux_logits = aux_logits_flat.view(b, s, -1)

        return {"fracture_logits": fracture_logits, "aux_logits": aux_logits}
