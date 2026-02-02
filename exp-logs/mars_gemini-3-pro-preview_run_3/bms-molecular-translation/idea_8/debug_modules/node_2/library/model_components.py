import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Splits an image into patches and embeds them.

    This layer transforms the input image (B, C, H, W) into a sequence of
    flattened patch embeddings (B, N, D), where N is the number of patches
    and D is the embedding dimension.
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use a convolution with kernel_size=patch_size and stride=patch_size
        # to effectively extract non-overlapping patches and project them.
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        # x after proj: (B, D, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, D, N)
        x = x.flatten(2)
        # Transpose to (B, N, D) for MLP processing
        x = x.transpose(1, 2)
        return x


class MlpBlock(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) block.
    Structure: Linear -> GELU -> Dropout -> Linear -> Dropout.
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """
    The core MLP-Mixer block consisting of Token-Mixing and Channel-Mixing layers.

    1. Token-Mixing: Mixes information across spatial locations (patches).
    2. Channel-Mixing: Mixes information across features (channels).
    """

    def __init__(
        self, num_tokens, dim, token_mixing_dim, channel_mixing_dim, dropout=0.0
    ):
        super().__init__()

        # Token Mixing
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixing = MlpBlock(
            in_features=num_tokens,
            hidden_features=token_mixing_dim,
            out_features=num_tokens,
            dropout=dropout,
        )

        # Channel Mixing
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixing = MlpBlock(
            in_features=dim,
            hidden_features=channel_mixing_dim,
            out_features=dim,
            dropout=dropout,
        )

    def forward(self, x):
        # x: (B, N, D)

        # --- Token Mixing ---
        y = self.norm1(x)
        # Transpose to (B, D, N) to apply MLP across tokens
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        # Transpose back to (B, N, D)
        y = y.transpose(1, 2)
        # Residual connection
        x = x + y

        # --- Channel Mixing ---
        y = self.norm2(x)
        # Apply MLP across channels (features) directly
        y = self.channel_mixing(y)
        # Residual connection
        x = x + y

        return x


class Attention(nn.Module):
    """
    Additive Attention (Bahdanau style) mechanism for the decoder.
    Calculates weights over encoder outputs based on the current decoder hidden state.
    """

    def __init__(self, enc_dim, dec_dim, attention_dim):
        super().__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attention_dim = attention_dim

        # Project encoder outputs: (B, N, enc_dim) -> (B, N, attn_dim)
        self.encoder_att = nn.Linear(enc_dim, attention_dim)

        # Project decoder hidden state: (B, dec_dim) -> (B, attn_dim)
        self.decoder_att = nn.Linear(dec_dim, attention_dim)

        # Calculate scores: (B, N, attn_dim) -> (B, N, 1)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: Previous decoder hidden state (B, dec_dim)
            encoder_outputs: Output from encoder (B, N, enc_dim)

        Returns:
            context: Weighted sum of encoder outputs (B, enc_dim)
            alpha: Attention weights (B, N, 1)
        """
        # Calculate attention scores
        # (B, N, attn_dim)
        projected_encoder = self.encoder_att(encoder_outputs)

        # (B, attn_dim) -> (B, 1, attn_dim) to broadcast across N
        projected_decoder = self.decoder_att(hidden).unsqueeze(1)

        # Combine and apply non-linearity
        # (B, N, attn_dim)
        combined_states = self.relu(projected_encoder + projected_decoder)

        # Calculate raw scores
        # (B, N, 1)
        attention_scores = self.full_att(combined_states)

        # Normalize scores to probabilities
        alpha = self.softmax(attention_scores)

        # Compute weighted context vector
        # (B, N, 1) * (B, N, enc_dim) -> (B, N, enc_dim) --sum--> (B, enc_dim)
        context = (alpha * encoder_outputs).sum(dim=1)

        return context, alpha
