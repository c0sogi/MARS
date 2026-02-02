import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).

    This module conditions the normalization of the input sequence on an auxiliary
    attribute vector (e.g., chemical formula counts). It predicts the scale (gamma)
    and shift (beta) parameters dynamically based on the condition.
    """

    def __init__(self, embed_dim, cond_dim):
        """
        Args:
            embed_dim (int): The dimension of the input sequence embeddings.
            cond_dim (int): The dimension of the conditioning attribute vector.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Standard LayerNorm without learnable affine parameters,
        # as we will predict them from the condition.
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        # Linear projection from condition vector to 2 * embed_dim (gamma + beta)
        self.proj = nn.Linear(cond_dim, 2 * embed_dim)

        # Initialize projection weights to zero so that the layer initially
        # acts as a standard identity LayerNorm (gamma=0, beta=0).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            condition (torch.Tensor): Conditioning vector of shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Modulated normalized tensor.
        """
        # Project condition to get modulation parameters
        # params shape: (batch_size, 2 * embed_dim)
        params = self.proj(condition)

        # Split into gamma (scale) and beta (shift)
        # gamma, beta shape: (batch_size, embed_dim)
        gamma, beta = params.chunk(2, dim=-1)

        # Expand dimensions to broadcast over the sequence length
        # shape becomes: (batch_size, 1, embed_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply normalization
        x_norm = self.norm(x)

        # Apply modulation: x * (1 + gamma) + beta
        # Using (1 + gamma) ensures that if gamma is 0, we retain the original scale.
        return x_norm * (1 + gamma) + beta


class AdaLNDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Adaptive Layer Normalization.

    Uses AdaLN to inject global attribute information into the decoding process
    at every sub-layer (Self-Attention, Cross-Attention, FFN).
    """

    def __init__(
        self, embed_dim, cond_dim, num_heads, ff_dim, dropout=0.1, encoder_dim=None
    ):
        """
        Args:
            embed_dim (int): Dimension of the decoder embeddings.
            cond_dim (int): Dimension of the conditioning vector.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network hidden layer.
            dropout (float): Dropout probability.
            encoder_dim (int, optional): Dimension of the encoder output.
                                         If None, assumes same as embed_dim.
        """
        super().__init__()

        # 1. Self-Attention (Masked)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = AdaLN(embed_dim, cond_dim)

        # 2. Cross-Attention (Encoder-Decoder)
        # Handle potential dimension mismatch between Encoder (e.g. EfficientNet 1280) and Decoder (256)
        kdim = encoder_dim if encoder_dim is not None else embed_dim
        vdim = encoder_dim if encoder_dim is not None else embed_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
            batch_first=True,
        )
        self.norm2 = AdaLN(embed_dim, cond_dim)

        # 3. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm3 = AdaLN(embed_dim, cond_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, condition, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            x (torch.Tensor): Decoder input (batch, seq_len, embed_dim).
            enc_out (torch.Tensor): Encoder output (batch, enc_seq_len, encoder_dim).
            condition (torch.Tensor): Attribute vector (batch, cond_dim).
            tgt_mask (torch.Tensor, optional): Causal mask for self-attention.
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for input sequence.

        Returns:
            torch.Tensor: Output tensor (batch, seq_len, embed_dim).
        """
        # --- Sub-layer 1: Masked Self-Attention ---
        # Pre-Norm architecture: Norm -> Attn -> Add
        residual = x
        x_norm = self.norm1(x, condition)

        attn_output, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = residual + self.dropout(attn_output)

        # --- Sub-layer 2: Cross-Attention ---
        residual = x
        x_norm = self.norm2(x, condition)

        # Query comes from decoder (x), Key/Value come from encoder (enc_out)
        attn_output, _ = self.cross_attn(query=x_norm, key=enc_out, value=enc_out)
        x = residual + self.dropout(attn_output)

        # --- Sub-layer 3: Feed Forward ---
        residual = x
        x_norm = self.norm3(x, condition)

        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)

        return x
