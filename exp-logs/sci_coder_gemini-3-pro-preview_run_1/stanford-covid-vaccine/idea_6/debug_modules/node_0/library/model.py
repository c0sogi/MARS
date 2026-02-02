import torch
import torch.nn as nn
from library.config import Config


class ResNetBlock1D(nn.Module):
    """
    A 1D Residual Block consisting of two convolutional layers with
    Batch Normalization and ReLU activation.
    """

    def __init__(self, channels, kernel_size, dropout_p):
        super(ResNetBlock1D, self).__init__()
        # Padding is calculated to maintain sequence length (same padding)
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class HybridResNetBiGRU(nn.Module):
    """
    Hybrid architecture combining 1D-ResNet for local feature extraction
    and Bidirectional GRU for global context aggregation.
    """

    def __init__(self):
        super(HybridResNetBiGRU, self).__init__()

        # 1. Embeddings
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.struct_embed = nn.Embedding(Config.VOCAB_SIZE_STRUCT, Config.EMBED_DIM)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)

        # Calculate concatenated embedding dimension
        input_dim = 3 * Config.EMBED_DIM

        # 2. Input Projection (Project to ResNet Channel dimensions)
        self.input_proj = nn.Conv1d(input_dim, Config.RESNET_CHANNELS, kernel_size=1)

        # 3. ResNet Backbone (Local Context)
        resnet_layers = []
        for _ in range(Config.RESNET_BLOCKS):
            resnet_layers.append(
                ResNetBlock1D(
                    channels=Config.RESNET_CHANNELS,
                    kernel_size=Config.RESNET_KERNEL_SIZE,
                    dropout_p=Config.DROPOUT,
                )
            )
        self.resnet = nn.Sequential(*resnet_layers)

        # 4. Bidirectional GRU (Global Context)
        self.gru = nn.GRU(
            input_size=Config.RESNET_CHANNELS,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        # 5. Output Head
        gru_output_dim = (
            Config.GRU_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.GRU_HIDDEN_SIZE
        )
        self.head = nn.Linear(gru_output_dim, Config.NUM_TARGETS)

    def forward(self, sequence, structure, predicted_loop_type):
        """
        Forward pass of the model.

        Args:
            sequence (torch.Tensor): (Batch, Seq_Len)
            structure (torch.Tensor): (Batch, Seq_Len)
            predicted_loop_type (torch.Tensor): (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, Num_Targets)
        """
        # 1. Embed Inputs
        emb_seq = self.seq_embed(sequence)  # (B, L, Embed)
        emb_struct = self.struct_embed(structure)  # (B, L, Embed)
        emb_loop = self.loop_embed(predicted_loop_type)  # (B, L, Embed)

        # Concatenate embeddings
        x = torch.cat([emb_seq, emb_struct, emb_loop], dim=2)  # (B, L, 3*Embed)

        # 2. ResNet Processing (Requires Channel-First format)
        # Permute to (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)

        # Project and pass through ResNet blocks
        x = self.input_proj(x)
        x = self.resnet(x)

        # 3. GRU Processing (Requires Sequence-First/Batch-First format)
        # Permute back to (Batch, Seq_Len, Channels)
        x = x.permute(0, 2, 1)

        # Pass through GRU
        # Output shape: (Batch, Seq_Len, Hidden*Directions)
        x, _ = self.gru(x)

        # 4. Output Projection
        out = self.head(x)  # (Batch, Seq_Len, Num_Targets)

        return out
