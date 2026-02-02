import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Transpose to (1, max_len, d_model) for batch_first broadcasting
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CNN_Tokenizer(nn.Module):
    """
    1D Convolutional Tokenizer.
    Projects raw one-hot features to d_model and aggregates local context.
    """

    def __init__(self, in_channels, d_model, kernel_size):
        super(CNN_Tokenizer, self).__init__()

        # Calculate padding to maintain sequence length (assuming odd kernel size)
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x input shape: (Batch, SeqLen, Channels)

        # Permute for Conv1d: (Batch, Channels, SeqLen)
        x = x.permute(0, 2, 1)

        x = self.conv(x)

        # Permute back: (Batch, SeqLen, d_model)
        x = x.permute(0, 2, 1)

        x = self.act(x)
        x = self.norm(x)
        return x


class ConvTransformer(nn.Module):
    """
    Signal-Filtered Convolutional Transformer.
    Consists of a CNN Tokenizer, Positional Encoding, Transformer Encoder, and a Regression Head.
    """

    def __init__(self):
        super(ConvTransformer, self).__init__()

        # Load hyperparameters from Config
        in_channels = Config.INPUT_CHANNELS
        d_model = Config.D_MODEL
        kernel_size = Config.KERNEL_SIZE
        nhead = Config.NHEAD
        num_layers = Config.NUM_LAYERS
        dim_feedforward = Config.DIM_FEEDFORWARD
        dropout = Config.DROPOUT
        num_targets = Config.NUM_TARGETS
        seq_len = Config.SEQ_LEN

        # 1. Tokenizer (Local Feature Extraction)
        self.tokenizer = CNN_Tokenizer(in_channels, d_model, kernel_size)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 50)

        # 3. Transformer Backbone (Global Context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN often stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Prediction Head
        self.head = nn.Linear(d_model, num_targets)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, SeqLen, Channels)
        Returns:
            output: Prediction tensor of shape (Batch, SeqLen, NumTargets)
        """
        # Tokenize: (B, L, C) -> (B, L, D)
        x = self.tokenizer(x)

        # Add Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Project to targets
        output = self.head(x)

        return output
