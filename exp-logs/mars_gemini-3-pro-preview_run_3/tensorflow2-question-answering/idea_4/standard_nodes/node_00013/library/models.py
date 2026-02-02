import torch
import torch.nn as nn
from library.config import Config


class TransformerEncoderBlock(nn.Module):
    """
    A lightweight Transformer Encoder block consisting of Multi-Head Self-Attention
    and a Feed-Forward Network.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Embed_Dim)
        # mask: (Batch, Seq_Len) - True where padding exists

        # Multi-Head Attention
        # Note: key_padding_mask expects True for positions to be ignored
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class SiameseRanker(nn.Module):
    """
    Siamese Self-Attention Ranker.
    Encodes Question and Candidate Paragraph independently using a shared Transformer block.
    Computes similarity via dot product of Global Max Pooled representations.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=config.MAX_VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=1,  # Assuming 1 is PAD based on data_utils
        )

        # Learnable Positional Encoding
        # We size it to the maximum possible length the ranker will see
        max_seq_len = max(config.MAX_Q_LEN, config.MAX_DOC_LEN)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, config.EMBEDDING_DIM)
        )

        # Shared Encoder Block
        self.encoder = TransformerEncoderBlock(
            embed_dim=config.EMBEDDING_DIM,
            num_heads=config.RANKER_HEADS,
            ff_dim=config.RANKER_FF_DIM,
            dropout=config.RANKER_DROPOUT,
        )

    def forward_one(self, ids):
        # ids: (Batch, Seq_Len)
        B, L = ids.shape

        # Create padding mask (True for pad tokens)
        # 1 is PAD_TOKEN_ID
        mask = ids == 1

        # Embeddings
        x = self.embedding(ids)  # (B, L, D)

        # Add Positional Encodings
        # Slice positional embeddings to current sequence length
        pos = self.pos_embedding[:, :L, :]
        x = x + pos

        # Transformer Encoding
        x = self.encoder(x, mask=mask)

        # Global Max Pooling
        # We need to mask out padding tokens so they don't affect max
        # Set padded positions to -inf
        x_masked = x.clone()
        # Expand mask to (B, L, D) for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand_as(x_masked)
        x_masked[mask_expanded] = -float("inf")

        # Max pool over sequence length dimension
        pooled_rep, _ = torch.max(x_masked, dim=1)  # (B, D)

        return pooled_rep

    def forward(self, q_ids, cand_ids):
        """
        Computes relevance score between question and candidate.
        """
        q_vec = self.forward_one(q_ids)  # (B, D)
        cand_vec = self.forward_one(cand_ids)  # (B, D)

        # Dot Product Similarity
        # (B, D) * (B, D) -> (B, D) -> sum -> (B,)
        scores = torch.sum(q_vec * cand_vec, dim=1)

        return scores


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block.
    Consists of Depthwise Conv -> Pointwise Conv -> Norm -> Activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()

        # Depthwise: Groups = in_channels
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding="same",  # Preserves length
        )

        # Pointwise: Kernel size = 1
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        residual = x

        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection if dimensions match
        if residual.shape[1] == out.shape[1]:
            out = out + residual

        return out


class SeparableConvReader(nn.Module):
    """
    Reader model using Depthwise Separable Convolutions to extract answer spans.
    Input is concatenated Question + Context.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.MAX_VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=1,
        )

        layers = []
        in_channels = config.EMBEDDING_DIM

        for _ in range(config.READER_LAYERS):
            layers.append(
                DepthwiseSeparableConv(
                    in_channels=in_channels,
                    out_channels=config.READER_FILTERS,
                    kernel_size=config.READER_KERNEL_SIZE,
                    dropout=config.READER_DROPOUT,
                )
            )
            in_channels = config.READER_FILTERS

        self.encoder = nn.Sequential(*layers)

        # Prediction Heads
        self.start_head = nn.Linear(config.READER_FILTERS, 1)
        self.end_head = nn.Linear(config.READER_FILTERS, 1)

    def forward(self, input_ids):
        # input_ids: (Batch, Seq_Len)

        x = self.embedding(input_ids)  # (B, L, D)

        # Conv1d expects (Batch, Channels, Length)
        x = x.transpose(1, 2)  # (B, D, L)

        # Pass through Conv Stack
        features = self.encoder(x)  # (B, Filters, L)

        # Prepare for linear layers: (B, L, Filters)
        features = features.transpose(1, 2)

        # Predict logits
        start_logits = self.start_head(features).squeeze(-1)  # (B, L)
        end_logits = self.end_head(features).squeeze(-1)  # (B, L)

        return start_logits, end_logits
