import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Normalizes the input by the root mean square of its values, then scales it.
    Does not apply mean centering, making it computationally efficient and
    preserving sign/magnitude information.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (..., dim)
        # Calculate RMS: sqrt(mean(x^2) + eps)
        var = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    Expects the input tensor to have a size of 2 * hidden_dim in the last dimension.
    It splits the input into two halves (a, b), applies Swish (SiLU) to a,
    and returns swish(a) * b.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split the input into two equal halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class RotaryEmbedding(nn.Module):
    """
    Computes Rotary Positional Embeddings (RoPE) frequencies.
    """

    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        # dim must be even for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos and sin
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        # Check if we need to recompute or extend the cache
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Concatenate to match the dimension of the head (repeat frequencies for real/imag parts logic)
            # This matches the rotate_half implementation [-x2, x1]
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()
            self.cached_sin = emb.sin()

        return self.cached_cos[:seq_len, :], self.cached_sin[:seq_len, :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to query and key tensors."""
    # q, k: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim)

    # Reshape cos/sin for broadcasting: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """
    Self-Attention mechanism with Rotary Positional Embeddings applied to Query and Key.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Project and reshape to (batch, seq_len, num_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Get RoPE embeddings
        cos, sin = self.rope(v, seq_len)

        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention computation: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot Product Attention
        # Scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask expected shape: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back and flatten: (batch, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        output = self.out_proj(attn_output)
        return output


class RoPETransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer that uses RoPEAttention.
    Supports both Pre-Norm and Post-Norm configurations.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        norm_first=False,
    ):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout=dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu" or activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, src, src_mask=None):
        x = src
        if self.norm_first:
            # Pre-Norm: x = x + sublayer(norm(x))
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-Norm: x = norm(x + sublayer(x))
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, mask):
        x = self.self_attn(x, mask=mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
