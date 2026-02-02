import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Encoder for Gesture Recognition.

    Consists of:
    1. Linear Projection of input features.
    2. Sinusoidal Positional Encoding.
    3. Stack of Transformer Encoder layers.
    4. Linear Classification Head.
    """

    def __init__(
        self,
        input_dim=85,
        num_classes=21,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    ):
        """
        Args:
            input_dim (int): Dimension of input features (default 85: 72 skeletal + 13 audio).
            num_classes (int): Number of output classes (default 21: 20 gestures + 1 background).
            d_model (int): The number of expected features in the encoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TemporalTransformer, self).__init__()

        self.d_model = d_model

        # 1. Input Projection
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Encoder
        # batch_first=True ensures input/output tensors are (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Classification Head
        self.decoder = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
            src_key_padding_mask: Tensor, shape [batch_size, seq_len].
                                  Boolean mask where True indicates padding positions
                                  (to be ignored by attention).

        Returns:
            output: Tensor, shape [batch_size, seq_len, num_classes]
        """
        # Project input features to d_model dimension
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through Transformer Encoder
        # src_key_padding_mask: If provided, True values are ignored in attention.
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # Project to class logits
        output = self.decoder(output)

        return output
