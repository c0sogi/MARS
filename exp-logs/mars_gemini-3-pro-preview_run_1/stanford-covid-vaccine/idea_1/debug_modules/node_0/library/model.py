import torch
import torch.nn as nn
from library.config import Config


class RNAConvNet(nn.Module):
    """
    1D Convolutional Neural Network for RNA degradation prediction.

    Architecture:
    - Embeddings for Sequence, Structure, and Loop Type
    - Concatenation of embeddings
    - Stack of 1D Convolutional Layers (Encoder)
    - Final 1D Convolutional Layer (Decoder/Head)
    """

    def __init__(self):
        super(RNAConvNet, self).__init__()

        # 1. Embeddings
        # We use the vocabulary sizes and embedding dimension from Config
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.struct_embedding = nn.Embedding(Config.VOCAB_SIZE_STRUCT, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)

        # Calculate the total number of input channels for the first Conv layer
        # 3 inputs * embedding dimension
        self.input_channels = 3 * Config.EMBED_DIM

        # 2. Encoder: Stack of Conv1d blocks
        layers = []
        in_channels = self.input_channels

        for _ in range(Config.LAYERS):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=Config.FILTER_CHANNELS,
                    kernel_size=Config.KERNEL_SIZE,
                    # Padding = (kernel_size - 1) // 2 ensures output length == input length for stride=1
                    padding=Config.KERNEL_SIZE // 2,
                    bias=False,  # Bias is redundant when using BatchNorm
                )
            )
            layers.append(nn.BatchNorm1d(Config.FILTER_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT))

            # The output of this layer becomes the input of the next
            in_channels = Config.FILTER_CHANNELS

        self.encoder = nn.Sequential(*layers)

        # 3. Decoder: Final projection to target columns
        # We predict all 5 targets as required by the submission format
        self.decoder = nn.Conv1d(
            in_channels=Config.FILTER_CHANNELS,
            out_channels=len(Config.TARGET_COLS),
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )

    def forward(self, sequence, structure, loop_type):
        """
        Forward pass of the network.

        Args:
            sequence (torch.Tensor): Shape (Batch, Seq_Len)
            structure (torch.Tensor): Shape (Batch, Seq_Len)
            loop_type (torch.Tensor): Shape (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5)
        """
        # Embed inputs: (Batch, Seq_Len) -> (Batch, Seq_Len, Embed_Dim)
        emb_seq = self.seq_embedding(sequence)
        emb_struct = self.struct_embedding(structure)
        emb_loop = self.loop_embedding(loop_type)

        # Concatenate along the feature dimension (dim=2)
        # Result: (Batch, Seq_Len, 3 * Embed_Dim)
        x = torch.cat([emb_seq, emb_struct, emb_loop], dim=2)

        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # Pass through Encoder
        # Result: (Batch, Filter_Channels, Seq_Len)
        x = self.encoder(x)

        # Pass through Decoder
        # Result: (Batch, 5, Seq_Len)
        x = self.decoder(x)

        # Permute back to (Batch, Seq_Len, Channels) for output
        x = x.permute(0, 2, 1)

        return x
