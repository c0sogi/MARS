import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class VolumetricTransformer(nn.Module):
    """
    2.5D Volumetric Sequence Transformer.

    Architecture:
    1. Time-Distributed CNN (EfficientNet-B0) to extract features from each slice.
    2. Linear Projection to reduce feature dimension.
    3. Positional Encoding + CLS Token injection.
    4. Transformer Encoder to aggregate information across the Z-axis (slices).
    5. Classification Head on the CLS token output.
    """

    def __init__(self):
        super(VolumetricTransformer, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # num_classes=0 removes the classifier, returning the pooled features
        # in_chans=3 matches our input (FLAIR, T1wCE, T2w)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED_BACKBONE,
            num_classes=0,
            in_chans=Config.NUM_CHANNELS,
        )

        # Get the feature dimension of the backbone (1280 for EfficientNet-B0)
        self.backbone_dim = self.backbone.num_features

        # 2. Projection Layer
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.ReLU(),
        )

        # 3. Sequence Components
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, Config.HIDDEN_DIM))

        # Learnable Positional Embeddings
        # Sequence length + 1 for the CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, Config.NUM_SLICES + 1, Config.HIDDEN_DIM)
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.HIDDEN_DIM * 4,
            dropout=Config.DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm usually stabilizes training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(Config.HIDDEN_DIM), nn.Linear(Config.HIDDEN_DIM, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize CLS token and Positional Embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initialize Linear layers in projector and classifier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Sequence, Channels, Height, Width)
        Returns:
            logits: Output tensor of shape (Batch, 1)
        """
        b, s, c, h, w = x.shape

        # --- 1. Time-Distributed Backbone ---
        # Reshape to (Batch * Sequence, C, H, W) to process all slices at once
        x = x.view(b * s, c, h, w)

        # Extract features: (B*S, Backbone_Dim)
        features = self.backbone(x)

        # --- 2. Projection & Reshaping ---
        # Reshape back to (Batch, Sequence, Backbone_Dim)
        features = features.view(b, s, self.backbone_dim)

        # Project to hidden dim: (Batch, Sequence, Hidden_Dim)
        features = self.projector(features)

        # --- 3. Sequence Preparation ---
        # Expand CLS token to match batch size: (Batch, 1, Hidden_Dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)

        # Concatenate CLS token to the front: (Batch, Sequence + 1, Hidden_Dim)
        x = torch.cat((cls_tokens, features), dim=1)

        # Add Positional Embeddings
        x = x + self.pos_embedding

        # --- 4. Transformer ---
        # Pass through Transformer Encoder
        x = self.transformer(x)

        # --- 5. Classification ---
        # Extract the output of the CLS token (index 0)
        cls_output = x[:, 0, :]

        # Predict logits
        logits = self.classifier(cls_output)

        return logits
