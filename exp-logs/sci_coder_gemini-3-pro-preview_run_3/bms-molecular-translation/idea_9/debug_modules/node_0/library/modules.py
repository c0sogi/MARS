import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    Consists of two linear layers with a GELU activation and dropout in between.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    """
    MLP-Mixer Block consisting of Token-Mixing and Channel-Mixing layers.
    This block allows the model to mix information across spatial locations (tokens)
    and across feature channels independently.
    """

    def __init__(self, dim, num_patches, token_dim, channel_dim, drop=0.0):
        super().__init__()

        # Token mixing: acts on the num_patches dimension
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixing = Mlp(
            in_features=num_patches,
            hidden_features=token_dim,
            out_features=num_patches,
            drop=drop,
        )

        # Channel mixing: acts on the dim dimension
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixing = Mlp(
            in_features=dim, hidden_features=channel_dim, out_features=dim, drop=drop
        )

    def forward(self, x):
        # x shape: (B, num_patches, dim)

        # Token mixing path
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, dim, num_patches)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)  # (B, num_patches, dim)
        x = x + y  # Skip connection

        # Channel mixing path
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y  # Skip connection

        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    Splits the image into fixed-size patches and linearly projects them.
    Implemented using a Conv2d layer with stride equal to patch size.
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the Transformer Decoder.
    Injects information about the relative or absolute position of tokens in the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter) so it's part of state_dict but not optimized
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, Seq_Len, D)
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    Wrapper around PyTorch's TransformerDecoderLayer.
    Configured to use batch_first=True to align with the rest of the pipeline.
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Args:
            tgt: (B, T, C) - Target sequence embeddings (InChI tokens)
            memory: (B, S, C) - Encoder output (Image patches)
            tgt_mask: (T, T) - Mask for self-attention (causal mask to prevent seeing future)
            tgt_key_padding_mask: (B, T) - Mask for target padding
            memory_key_padding_mask: (B, S) - Mask for memory padding
        """
        return self.decoder_layer(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
