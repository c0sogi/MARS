import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HistogramBinning(nn.Module):
    """
    Computes the interaction matrix between query and document embeddings and
    discretizes the cosine similarities into a fixed number of bins.
    """

    def __init__(self, num_bins=Config.HISTOGRAM_BINS, min_sim=-1.0, max_sim=1.0):
        super(HistogramBinning, self).__init__()
        self.num_bins = num_bins

        # Create bin centers uniformly spaced between min_sim and max_sim
        # Example: 11 bins from -1 to 1 -> [-1.0, -0.8, ..., 1.0]
        step = (max_sim - min_sim) / (num_bins - 1)
        centers = [min_sim + i * step for i in range(num_bins)]

        # Register centers as a buffer so it's part of the state_dict but not a parameter
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))

        # Sigma controls the width of the Gaussian kernel for soft binning
        # Setting sigma to half the step size provides reasonable overlap
        self.sigma = step / 2.0

    def forward(self, query_embeds, doc_embeds):
        """
        Args:
            query_embeds: Tensor of shape (batch_size, q_len, embed_dim)
            doc_embeds: Tensor of shape (batch_size, d_len, embed_dim)

        Returns:
            histograms: Tensor of shape (batch_size, q_len, num_bins)
        """
        # Normalize embeddings to ensure dot product equals cosine similarity
        q_norm = F.normalize(query_embeds, p=2, dim=-1)
        d_norm = F.normalize(doc_embeds, p=2, dim=-1)

        # Compute Cosine Similarity Matrix
        # (batch, q_len, dim) x (batch, dim, d_len) -> (batch, q_len, d_len)
        sim_matrix = torch.bmm(q_norm, d_norm.transpose(1, 2))

        # Prepare for broadcasting against bin centers
        # sim_matrix: (batch, q_len, d_len, 1)
        s = sim_matrix.unsqueeze(-1)
        # centers: (1, 1, 1, num_bins)
        mu = self.centers.view(1, 1, 1, -1)

        # Compute Gaussian RBF (Soft Binning)
        # Result: (batch, q_len, d_len, num_bins)
        rbf = torch.exp(-torch.pow(s - mu, 2) / (2 * self.sigma**2))

        # Sum over the document dimension to get the matching density for each query word
        # Result: (batch, q_len, num_bins)
        histograms = torch.sum(rbf, dim=2)

        return histograms


class QRNNLayer(nn.Module):
    """
    Implements a Quasi-Recurrent Neural Network (QRNN) layer.
    Uses 1D convolution for feature extraction and dynamic pooling for recurrence.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size=Config.QRNN_KERNEL_SIZE,
        dropout=Config.QRNN_DROPOUT,
    ):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Padding is set to maintain sequence length (assuming odd kernel size)
        self.pad = kernel_size // 2

        # Convolution produces 3 outputs per hidden unit: Candidate (Z), Forget (F), Output (O)
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size * 3,
            kernel_size=kernel_size,
            padding=self.pad,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)

        Returns:
            h: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        # Transpose for Conv1d: (batch, input_size, seq_len)
        x_t = x.transpose(1, 2)

        # Apply Convolution
        # Output: (batch, hidden_size * 3, seq_len)
        gates = self.conv(x_t)

        # Ensure output length matches input length (handle potential padding edge cases)
        if gates.size(2) > seq_len:
            gates = gates[:, :, :seq_len]

        # Transpose back: (batch, seq_len, hidden_size * 3)
        gates = gates.transpose(1, 2)

        # Split into Z (candidate), F (forget), O (output)
        z, f, o = torch.split(gates, self.hidden_size, dim=2)

        # Apply activation functions
        z = torch.tanh(z)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        # Dynamic Pooling (Recurrence)
        # h_t = f_t * h_{t-1} + (1 - f_t) * z_t

        h_list = []
        # Initialize previous hidden state as zeros
        h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Iterate through time steps
        for t in range(seq_len):
            z_t = z[:, t, :]
            f_t = f[:, t, :]

            # Recurrence relation
            h_t = f_t * h_prev + (1 - f_t) * z_t

            h_prev = h_t
            h_list.append(h_t)

        # Stack results: (batch, seq_len, hidden_size)
        h = torch.stack(h_list, dim=1)

        # Apply output gate
        h = h * o

        # Apply dropout
        h = self.dropout(h)

        return h
