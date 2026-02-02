import torch
import torch.nn as nn
import math
import library.config as config


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape for batch_first=True: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Dim)
        """
        # Slice pe to the current sequence length
        # self.pe is (1, max_len, d_model), x is (Batch, Seq_Len, d_model)
        # Broadcasting handles the batch dimension
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HybridCNNTransformer(nn.Module):
    """
    A hybrid architecture combining CNN for local feature extraction and
    Transformer for global context modeling.
    """

    def __init__(
        self,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        cnn_filters=config.CNN_FILTERS,
        cnn_kernel_size=config.CNN_KERNEL_SIZE,
        transformer_layers=config.TRANSFORMER_LAYERS,
        num_heads=config.NUM_HEADS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.DROPOUT,
        num_classes=config.TOP_K_TAGS,
        max_len=config.MAX_LEN,
    ):
        super(HybridCNNTransformer, self).__init__()

        # 1. Embedding Layer
        # padding_idx=0 ensures the padding token vector remains zero and is not updated
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. Local Feature Extraction (CNN)
        # Calculate padding to keep sequence length same for odd kernel sizes
        # padding = (k - 1) // 2
        padding = (cnn_kernel_size - 1) // 2
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

        # 3. Global Context Modeling (Transformer)
        # Positional Encoding is applied to the features before the Transformer
        self.pos_encoder = PositionalEncoding(
            d_model=cnn_filters, max_len=max_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_filters,
            nhead=num_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,  # Input/Output is (Batch, Seq, Dim)
            norm_first=True,  # Pre-LayerNorm often stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # 4. Output Layer
        self.classifier = nn.Linear(cnn_filters, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input token IDs of shape (Batch, Seq_Len).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Create mask for padding tokens (True where padding exists)
        # x is (Batch, Seq_Len)
        key_padding_mask = x == 0

        # 1. Embedding
        # Output: (Batch, Seq_Len, Embed_Dim)
        x = self.embedding(x)

        # 2. CNN
        # PyTorch Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        # Permute back to (Batch, Seq_Len, Channels/Filters) for Transformer
        x = x.permute(0, 2, 1)

        # 3. Transformer
        # Add positional information to the CNN features
        x = self.pos_encoder(x)

        # Pass through Transformer Encoder
        # src_key_padding_mask prevents attention to padding tokens
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # 4. Global Average Pooling (Masked)
        # We want to average the features over the sequence dimension, ignoring padding.

        # Invert mask to get valid positions (1 for valid, 0 for pad)
        # mask shape: (Batch, Seq_Len, 1)
        mask = (~key_padding_mask).unsqueeze(-1).float()

        # Zero out padding positions
        x = x * mask

        # Sum features over sequence dimension
        sum_features = x.sum(dim=1)  # (Batch, CNN_Filters)

        # Count valid tokens per sequence
        valid_lengths = mask.sum(dim=1)  # (Batch, 1)
        valid_lengths = torch.clamp(valid_lengths, min=1e-9)  # Avoid division by zero

        # Compute average
        x = sum_features / valid_lengths

        # 5. Classifier
        # Output logits (BCEWithLogitsLoss will handle Sigmoid)
        logits = self.classifier(x)

        return logits
