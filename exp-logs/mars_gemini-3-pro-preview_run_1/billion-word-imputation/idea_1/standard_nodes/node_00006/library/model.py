import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedConvBlock(nn.Module):
    """
    A Gated Convolutional Block consisting of a 1D convolution,
    Gated Linear Unit (GLU), dropout, and a residual connection.
    """

    def __init__(self, hidden_dim, kernel_size, dropout):
        super().__init__()

        # Calculate padding to maintain sequence length
        # We assume kernel_size is odd (e.g., 3, 5)
        padding = (kernel_size - 1) // 2

        # Convolution maps hidden_dim -> 2 * hidden_dim for GLU
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Hidden_Dim, Seq_Len)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Hidden_Dim, Seq_Len)
        """
        residual = x

        # Convolution
        out = self.conv(x)

        # Gated Linear Unit (splits channels in half: A * sigmoid(B))
        out = F.glu(out, dim=1)

        # Dropout
        out = self.dropout(out)

        # Residual connection
        return residual + out


class GatedInfillingModel(nn.Module):
    """
    Gated Convolutional Sequence Labeler for Sentence Infilling.
    Predicts the probability distribution of the missing word for every inter-word gap.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=Config.KERNEL_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        padding_idx=0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Input Projection (if embedding dim differs from hidden dim)
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Stack of Gated Convolutional Blocks
        self.encoder = nn.ModuleList(
            [
                GatedConvBlock(hidden_dim, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output Projection to Vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (Batch, Seq_Len)

        Returns:
            torch.Tensor: Logits of shape (Batch, Seq_Len, Vocab_Size)
        """
        # 1. Embedding Lookup
        # Shape: (Batch, Seq_Len, Embed_Dim)
        x = self.embedding(input_ids)

        # 2. Project to Hidden Dimension
        # Shape: (Batch, Seq_Len, Hidden_Dim)
        x = self.input_proj(x)

        # 3. Transpose for Conv1d (Channels first)
        # Shape: (Batch, Hidden_Dim, Seq_Len)
        x = x.transpose(1, 2)

        # 4. Pass through Gated Conv Blocks
        for layer in self.encoder:
            x = layer(x)

        # 5. Transpose back (Channels last)
        # Shape: (Batch, Seq_Len, Hidden_Dim)
        x = x.transpose(1, 2)

        # 6. Output Projection
        # Shape: (Batch, Seq_Len, Vocab_Size)
        logits = self.output_proj(x)

        return logits
