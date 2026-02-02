import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DualPathTransformer(nn.Module):
    """
    Dual-Path Transformer-Fused Network.

    Path 1 (Context): EfficientNet-B0 -> Projection -> Transformer -> Context Vector
    Path 2 (Linear): Baseline FVC + Weeks -> Skip Connection
    Fusion: Concat(Context, Linear) -> MLP -> Output
    """

    def __init__(self):
        super(DualPathTransformer, self).__init__()

        self.embed_dim = Config.EMBED_DIM

        # ==========================
        # Path 1: Context Backbone
        # ==========================
        # Frozen EfficientNet-B0
        # num_classes=0 returns the pooled feature vector (1280 dim for b0)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_dim = self.backbone.num_features  # 1280

        # Projection for images: 1280 -> 128
        self.img_proj = nn.Linear(self.backbone_dim, self.embed_dim)

        # Learnable Positional Embeddings for the 3 slices (Top, Mid, Bottom)
        # Shape: (1, 3, embed_dim) to broadcast over batch
        self.slice_pos_embed = nn.Parameter(torch.randn(1, 3, self.embed_dim))

        # ==========================
        # Path 1: Metadata Token
        # ==========================
        # Embeddings for categoricals
        self.sex_embed = nn.Embedding(2, self.embed_dim)  # Male=0, Female=1
        self.smoke_embed = nn.Embedding(3, self.embed_dim)  # Ex=0, Never=1, Current=2

        # Projection for continuous Age
        self.age_proj = nn.Linear(1, self.embed_dim)

        # ==========================
        # Path 1: Transformer
        # ==========================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # ==========================
        # Fusion & Head
        # ==========================
        # Input to head:
        #   Flattened Context: (3 Image Tokens + 1 Meta Token) * embed_dim = 4 * 128 = 512
        #   Linear Features: 2 (Baseline FVC + Weeks)
        self.context_flat_dim = 4 * self.embed_dim
        self.linear_feat_dim = 2

        fusion_input_dim = self.context_flat_dim + self.linear_feat_dim

        # MLP Head: Linear -> ReLU -> Linear
        # Hidden dimension arbitrary, choosing 128
        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output: FVC (mu), Confidence (sigma)
        )

    def forward(self, images, meta_age, meta_sex, meta_smoke, linear_features):
        """
        Args:
            images: (B, 3, H, W)
            meta_age: (B,)
            meta_sex: (B,)
            meta_smoke: (B,)
            linear_features: (B, 2)
        """
        batch_size = images.shape[0]

        # ==========================
        # Path 1: Image Processing
        # ==========================
        # Reshape for backbone: (B*3, 1, H, W) -> (B*3, 3, H, W)
        # EfficientNet expects 3 channels. We repeat the grayscale channel.
        x_img = images.view(batch_size * 3, 1, Config.IMG_SIZE, Config.IMG_SIZE)
        x_img = x_img.repeat(1, 3, 1, 1)

        # Extract features
        # (B*3, 1280)
        features = self.backbone(x_img)

        # Reshape back to sequence: (B, 3, 1280)
        features = features.view(batch_size, 3, -1)

        # Project to embed_dim: (B, 3, 128)
        img_tokens = self.img_proj(features)

        # Add Positional Embeddings
        img_tokens = img_tokens + self.slice_pos_embed

        # ==========================
        # Path 1: Metadata Processing
        # ==========================
        # Create Metadata Token by summing embeddings
        # Reshape age for linear layer: (B, 1)
        age_emb = self.age_proj(meta_age.unsqueeze(1))
        sex_emb = self.sex_embed(meta_sex)
        smoke_emb = self.smoke_embed(meta_smoke)

        # Sum to get single token: (B, 128)
        meta_token = age_emb + sex_emb + smoke_emb

        # Reshape to sequence format: (B, 1, 128)
        meta_token = meta_token.unsqueeze(1)

        # ==========================
        # Path 1: Transformer
        # ==========================
        # Concatenate tokens: [Image_Top, Image_Mid, Image_Bot, Meta]
        # Shape: (B, 4, 128)
        context_seq = torch.cat([img_tokens, meta_token], dim=1)

        # Pass through Transformer
        context_out = self.transformer(context_seq)

        # Flatten: (B, 512)
        context_flat = context_out.reshape(batch_size, -1)

        # ==========================
        # Fusion
        # ==========================
        # Concatenate Context with Linear Features (Skip Connection)
        # Shape: (B, 512 + 2)
        fused = torch.cat([context_flat, linear_features], dim=1)

        # ==========================
        # Output Head
        # ==========================
        out = self.head(fused)

        # Split output
        fvc_pred = out[:, 0]
        sigma_raw = out[:, 1]

        # Enforce positivity for sigma (Confidence)
        # softplus(x) + epsilon to avoid div by zero
        sigma_pred = F.softplus(sigma_raw) + 1e-3

        return fvc_pred, sigma_pred
