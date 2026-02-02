import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class SelfAttention(nn.Module):
    """
    Computes attention weights for a sequence of embeddings to create a single vector representation.
    """

    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for pad
        Returns:
            context_vector: (batch, input_dim)
            weights: (batch, seq_len, 1)
        """
        # (batch, seq_len, 1)
        scores = self.projection(embeddings)

        if mask is not None:
            # Mask padding positions with a large negative value
            scores = scores.squeeze(-1) + (1.0 - mask) * -1e9
            scores = scores.unsqueeze(-1)

        weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch, input_dim)
        context_vector = torch.sum(embeddings * weights, dim=1)

        return context_vector, weights


class ANBoWRanker(nn.Module):
    """
    Attention-Weighted Neural Bag-of-Words Ranker.
    """

    def __init__(self, embedding_matrix=None, freeze_embeddings=True):
        super(ANBoWRanker, self).__init__()

        # Embedding Layer
        if embedding_matrix is not None:
            vocab_size, emb_dim = embedding_matrix.shape
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            vocab_size = Config.VOCAB_SIZE
            emb_dim = Config.EMBEDDING_DIM
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.embedding_dim = emb_dim

        # Attention Layers
        self.q_attention = SelfAttention(emb_dim, Config.RANKER_HIDDEN_DIM)
        self.c_attention = SelfAttention(emb_dim, Config.RANKER_HIDDEN_DIM)

        # Interaction MLP
        # Input features: [q_vec, c_vec, |q-c|, q*c] -> 4 * emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(4 * emb_dim, Config.RANKER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.RANKER_DROPOUT),
            nn.Linear(Config.RANKER_HIDDEN_DIM, 1),
        )

    def forward(self, q_ids, c_ids):
        """
        Args:
            q_ids: (batch, q_len)
            c_ids: (batch, c_len)
        Returns:
            logits: (batch, )
        """
        # Create masks (assuming 0 is PAD)
        q_mask = (q_ids != 0).float()
        c_mask = (c_ids != 0).float()

        # Embeddings: (batch, seq_len, emb_dim)
        q_emb = self.embedding(q_ids)
        c_emb = self.embedding(c_ids)

        # Aggregate via Attention
        q_vec, _ = self.q_attention(q_emb, q_mask)
        c_vec, _ = self.c_attention(c_emb, c_mask)

        # Heuristic Matching
        diff = torch.abs(q_vec - c_vec)
        prod = q_vec * c_vec

        combined = torch.cat([q_vec, c_vec, diff, prod], dim=1)

        # Score
        logits = self.classifier(combined)
        return logits.squeeze(-1)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block for efficient sequence modeling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, length)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class BiDAFAttention(nn.Module):
    """
    Bi-Directional Attention Flow Layer.
    Computes Context-to-Query and Query-to-Context attention.
    """

    def __init__(self, hidden_dim):
        super(BiDAFAttention, self).__init__()
        # Similarity weight vector w (3 * hidden_dim because similarity input is [h; u; h*u])
        self.w = nn.Linear(3 * hidden_dim, 1, bias=False)

    def forward(self, context, query, c_mask, q_mask):
        """
        Args:
            context: (batch, c_len, dim)
            query: (batch, q_len, dim)
            c_mask: (batch, c_len)
            q_mask: (batch, q_len)
        Returns:
            output: (batch, c_len, 4*dim)
        """
        batch_size, c_len, dim = context.size()
        q_len = query.size(1)

        # Calculate Similarity Matrix S
        # Expand to (batch, c_len, q_len, dim)
        c_expand = context.unsqueeze(2).expand(-1, -1, q_len, -1)
        q_expand = query.unsqueeze(1).expand(-1, c_len, -1, -1)

        # Element-wise product
        elem_prod = c_expand * q_expand

        # Concatenate [c; q; c*q] -> (batch, c_len, q_len, 3*dim)
        cat_data = torch.cat([c_expand, q_expand, elem_prod], dim=-1)

        # Compute similarity scores S: (batch, c_len, q_len)
        S = self.w(cat_data).squeeze(-1)

        # Apply masking
        # q_mask: (batch, 1, q_len)
        # c_mask: (batch, c_len, 1)
        S_c2q = S + (1.0 - q_mask.unsqueeze(1)) * -1e9
        S_q2c = S + (1.0 - c_mask.unsqueeze(2)) * -1e9

        # Context-to-Query Attention (C2Q)
        # alpha: (batch, c_len, q_len)
        alpha = F.softmax(S_c2q, dim=-1)
        # a: (batch, c_len, dim) = alpha * query
        a = torch.bmm(alpha, query)

        # Query-to-Context Attention (Q2C)
        # m: (batch, c_len) -> max over query dim
        m, _ = torch.max(S_q2c, dim=-1)
        # beta: (batch, c_len) -> softmax over context dim
        beta = F.softmax(m, dim=-1)
        # c_tilde: (batch, dim) = beta * context
        c_tilde = torch.bmm(beta.unsqueeze(1), context).squeeze(1)
        # Tile c_tilde to match context length: (batch, c_len, dim)
        c_tilde_tiled = c_tilde.unsqueeze(1).expand(-1, c_len, -1)

        # Fusion
        # [c; a; c*a; c*c_tilde]
        output = torch.cat([context, a, context * a, context * c_tilde_tiled], dim=-1)

        return output


class ConvBiDAFReader(nn.Module):
    """
    Convolutional Bi-Directional Attention Flow Reader.
    Uses Conv1D for local context and Separable Convs for modeling.
    """

    def __init__(self, embedding_matrix=None, freeze_embeddings=True):
        super(ConvBiDAFReader, self).__init__()

        # Embedding Layer
        if embedding_matrix is not None:
            vocab_size, emb_dim = embedding_matrix.shape
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            vocab_size = Config.VOCAB_SIZE
            emb_dim = Config.EMBEDDING_DIM
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.embedding_dim = emb_dim

        # Local Context Layer (1D Conv)
        self.q_conv = nn.Conv1d(
            emb_dim,
            Config.READER_HIDDEN_DIM,
            kernel_size=Config.READER_CONV_KERNEL_SIZE,
            padding=Config.READER_CONV_KERNEL_SIZE // 2,
        )
        self.c_conv = nn.Conv1d(
            emb_dim,
            Config.READER_HIDDEN_DIM,
            kernel_size=Config.READER_CONV_KERNEL_SIZE,
            padding=Config.READER_CONV_KERNEL_SIZE // 2,
        )

        # BiDAF Layer
        self.bidaf = BiDAFAttention(Config.READER_HIDDEN_DIM)

        # Modeling Layer (Stack of Separable Convs)
        # BiDAF output dim is 4 * hidden_dim, project back to hidden_dim
        self.modeling_proj = nn.Linear(
            4 * Config.READER_HIDDEN_DIM, Config.READER_HIDDEN_DIM
        )

        self.modeling_convs = nn.Sequential(
            DepthwiseSeparableConv(
                Config.READER_HIDDEN_DIM,
                Config.READER_HIDDEN_DIM,
                kernel_size=5,
                dropout=Config.READER_DROPOUT,
            ),
            DepthwiseSeparableConv(
                Config.READER_HIDDEN_DIM,
                Config.READER_HIDDEN_DIM,
                kernel_size=5,
                dropout=Config.READER_DROPOUT,
            ),
        )

        # Output Heads
        self.start_head = nn.Linear(Config.READER_HIDDEN_DIM, 1)
        self.end_head = nn.Linear(Config.READER_HIDDEN_DIM, 1)

    def forward(self, q_ids, c_ids):
        """
        Args:
            q_ids: (batch, q_len)
            c_ids: (batch, c_len)
        Returns:
            start_logits: (batch, c_len)
            end_logits: (batch, c_len)
        """
        # Masks
        q_mask = (q_ids != 0).float()
        c_mask = (c_ids != 0).float()

        # Embed
        q_emb = self.embedding(q_ids)  # (B, Q, E)
        c_emb = self.embedding(c_ids)  # (B, C, E)

        # Local Context (Conv1d expects B, Channels, Length)
        q_emb_t = q_emb.transpose(1, 2)
        c_emb_t = c_emb.transpose(1, 2)

        q_ctx = self.q_conv(q_emb_t).transpose(1, 2)  # (B, Q, H)
        c_ctx = self.c_conv(c_emb_t).transpose(1, 2)  # (B, C, H)

        # BiDAF Interaction
        # Output: (B, C, 4*H)
        interaction = self.bidaf(c_ctx, q_ctx, c_mask, q_mask)

        # Modeling
        # Project down to hidden dim
        model_input = self.modeling_proj(interaction)  # (B, C, H)
        model_input_t = model_input.transpose(1, 2)  # (B, H, C)

        model_out_t = self.modeling_convs(model_input_t)
        model_out = model_out_t.transpose(1, 2)  # (B, C, H)

        # Output
        start_logits = self.start_head(model_out).squeeze(-1)  # (B, C)
        end_logits = self.end_head(model_out).squeeze(-1)  # (B, C)

        # Mask logits (set pad positions to -inf to prevent prediction in pad area)
        start_logits = start_logits + (1.0 - c_mask) * -1e9
        end_logits = end_logits + (1.0 - c_mask) * -1e9

        return start_logits, end_logits
