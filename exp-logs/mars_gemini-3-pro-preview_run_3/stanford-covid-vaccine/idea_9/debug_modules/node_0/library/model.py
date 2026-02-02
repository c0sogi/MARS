import torch
import torch.nn as nn


class InceptionStem(nn.Module):
    """
    Multi-Scale Inception Stem.
    Applies parallel 1D convolutions with different kernel sizes to capture
    local structural motifs at varying resolutions (single nucleotide, codon, etc.).
    """

    def __init__(self, in_channels, out_channels_per_branch, kernel_sizes):
        super(InceptionStem, self).__init__()
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            # padding='same' ensures the output sequence length matches the input length.
            # This is valid for odd kernel sizes with stride=1.
            self.branches.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels_per_branch,
                    kernel_size=k,
                    padding="same",
                )
            )

        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, In_Channels, Seq_Len).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Out_Channels * Branches, Seq_Len).
        """
        # Apply each branch in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate outputs along the channel dimension (dim=1)
        out = torch.cat(branch_outputs, dim=1)

        # Apply activation
        out = self.activation(out)
        return out


class RNAGRUModel(nn.Module):
    """
    Main RNA Degradation Prediction Model.
    Architecture:
    1. Multi-Scale Inception Stem (Projections)
    2. Bidirectional GRU Backbone (Contextualization)
    3. Linear Head (Prediction)
    """

    def __init__(self, config):
        super(RNAGRUModel, self).__init__()

        # --- 1. Inception Stem ---
        # Input: (Batch, Seq_Len, Feature_Dim) -> Permuted to (Batch, Feature_Dim, Seq_Len)
        self.stem = InceptionStem(
            in_channels=config.feature_dim,
            out_channels_per_branch=config.stem_channels,
            kernel_sizes=config.inception_kernels,
        )

        # Calculate total output channels from the stem
        # Total = stem_channels * number of branches
        self.stem_out_dim = config.stem_channels * len(config.inception_kernels)

        # --- 2. Backbone (BiGRU) ---
        # Input: (Batch, Seq_Len, Stem_Out_Dim)
        self.gru = nn.GRU(
            input_size=self.stem_out_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Calculate GRU output dimension
        # If bidirectional, output is hidden_dim * 2
        self.gru_out_dim = (
            config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        )

        # --- 3. Output Head ---
        # Projects to the 5 target columns
        self.head = nn.Linear(self.gru_out_dim, config.num_targets)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Feature_Dim).
        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, Num_Targets).
        """
        # Permute to (Batch, Feature_Dim, Seq_Len) for Conv1d operations in the Stem
        x = x.permute(0, 2, 1)

        # Pass through Inception Stem
        x = self.stem(x)

        # Permute back to (Batch, Seq_Len, Stem_Out_Dim) for RNN operations
        x = x.permute(0, 2, 1)

        # Pass through BiGRU Backbone
        # output shape: (Batch, Seq_Len, Hidden_Dim * Directions)
        x, _ = self.gru(x)

        # Pass through Linear Head
        # output shape: (Batch, Seq_Len, Num_Targets)
        out = self.head(x)

        return out
