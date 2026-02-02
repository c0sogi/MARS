import torch
import torch.nn as nn
from library.config import Config


class PeriodicEmbedding(nn.Module):
    """
    Learns periodic representations for continuous numerical features.
    Projects scalar inputs into a high-dimensional space using sine and cosine functions
    with learnable frequencies and offsets.
    """

    def __init__(self, num_features, num_frequencies, embedding_dim, sigma):
        super().__init__()
        self.num_features = num_features
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim

        # Initialize frequencies and offsets
        # Shape: (Num_Features, Num_Frequencies)
        # Frequencies initialized with normal distribution scaled by sigma
        self.frequencies = nn.Parameter(
            torch.randn(num_features, num_frequencies) * sigma
        )

        # Offsets initialized to zero
        self.offsets = nn.Parameter(torch.zeros(num_features, num_frequencies))

        # Projection layer: [sin, cos] -> embedding_dim
        # Input dim is 2 * num_frequencies because we concat sin and cos
        self.proj = nn.Linear(2 * num_frequencies, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch_Size, Num_Features)

        Returns:
            Tensor of shape (Batch_Size, Num_Features, Embedding_Dim)
        """
        # Expand x for broadcasting: (B, N, 1)
        x = x.unsqueeze(-1)

        # Calculate argument: w*x + b
        # frequencies/offsets broadcast to (B, N, F)
        args = x * self.frequencies + self.offsets

        # Compute periodic features
        sin_feat = torch.sin(args)
        cos_feat = torch.cos(args)

        # Concatenate: (B, N, 2*F)
        periodic_feats = torch.cat([sin_feat, cos_feat], dim=-1)

        # Project to model dimension
        out = self.proj(periodic_feats)

        return out


class ManufacturingTransformer(nn.Module):
    """
    Unified Transformer architecture for Manufacturing Control data.
    Combines numerical features (via Periodic Embeddings) and sequence features
    (via Entity Embeddings) into a single sequence processed by a Transformer Encoder.
    Uses a [CLS] token for final classification.
    """

    def __init__(self, num_numerical_features, vocab_size, seq_len):
        super().__init__()

        # Hyperparameters from Config
        embed_dim = Config.EMBED_DIM
        num_heads = Config.NUM_HEADS
        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT
        forward_expansion = Config.FORWARD_EXPANSION
        num_freqs = Config.NUM_FREQUENCIES
        sigma = Config.SIGMA
        head_hidden_dim = Config.HEAD_HIDDEN_DIM
        num_classes = Config.NUM_CLASSES

        # 1. Feature Embeddings
        self.num_embedding = PeriodicEmbedding(
            num_features=num_numerical_features,
            num_frequencies=num_freqs,
            embedding_dim=embed_dim,
            sigma=sigma,
        )

        self.seq_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        # 2. Special Tokens and Positional Encoding
        # [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable Positional Embeddings
        # Sequence structure: [CLS] + [Num Features] + [Seq Features]
        self.total_seq_len = 1 + num_numerical_features + seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_seq_len, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * forward_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN is generally more stable
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Readout Head
        # MLP applied to the [CLS] token output
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Standard transformer initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num, x_seq):
        """
        Args:
            x_num: (Batch, Num_Numerical_Features) - Float tensor
            x_seq: (Batch, Seq_Len) - Long tensor

        Returns:
            logits: (Batch, Num_Classes)
        """
        batch_size = x_num.size(0)

        # 1. Embed Inputs
        # Numerical: (B, N_num) -> (B, N_num, D)
        emb_num = self.num_embedding(x_num)

        # Sequence: (B, N_seq) -> (B, N_seq, D)
        emb_seq = self.seq_embedding(x_seq)

        # [CLS] Token: (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 2. Construct Sequence
        # Order: [CLS], [Numerical Features], [Sequence Features]
        x = torch.cat((cls_tokens, emb_num, emb_seq), dim=1)

        # 3. Add Positional Encodings
        # Add position embeddings (broadcasting over batch)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 4. Transformer Encoding
        # Output: (B, Total_Len, D)
        x = self.transformer_encoder(x)

        # 5. Readout
        # Extract [CLS] token state (index 0)
        cls_output = x[:, 0, :]

        # Pass through MLP head
        logits = self.head(cls_output)

        return logits
