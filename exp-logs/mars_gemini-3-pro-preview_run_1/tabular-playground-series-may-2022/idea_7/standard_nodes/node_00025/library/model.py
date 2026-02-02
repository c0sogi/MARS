import torch
import torch.nn as nn
from library.config import Config


class FeatureTokenizer(nn.Module):
    """
    Learns linear representations for continuous numerical features.
    Projects scalar inputs into a high-dimensional space using linear transformation:
    Embedding = Value * Weight + Bias
    Cite solution_lesson_node_00023
    """

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Weight: (Num_Features, Embedding_Dim)
        self.weights = nn.Parameter(torch.randn(num_features, embedding_dim))

        # Bias: (Num_Features, Embedding_Dim)
        self.bias = nn.Parameter(torch.zeros(num_features, embedding_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch_Size, Num_Features)

        Returns:
            Tensor of shape (Batch_Size, Num_Features, Embedding_Dim)
        """
        # x: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)

        # weights: (N, D) -> (1, N, D)
        w = self.weights.unsqueeze(0)

        # bias: (N, D) -> (1, N, D)
        b = self.bias.unsqueeze(0)

        # Linear tokenization: x * w + b
        return x * w + b


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
        head_hidden_dim = Config.HEAD_HIDDEN_DIM
        num_classes = Config.NUM_CLASSES

        # 1. Feature Embeddings
        # Use Linear Feature Tokenization (Cite solution_lesson_node_00023)
        self.num_embedding = FeatureTokenizer(
            num_features=num_numerical_features,
            embedding_dim=embed_dim,
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
