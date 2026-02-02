import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineInteraction(nn.Module):
    """
    Computes the cosine similarity matrix between query and document embeddings.
    This serves as the 'Interaction Matrix' input for the Ranking Module.
    """

    def __init__(self):
        super(CosineInteraction, self).__init__()

    def forward(self, query_embeddings, doc_embeddings):
        """
        Args:
            query_embeddings: Tensor of shape (batch_size, q_len, embedding_dim)
            doc_embeddings: Tensor of shape (batch_size, c_len, embedding_dim)

        Returns:
            interaction_matrix: Tensor of shape (batch_size, q_len, c_len) containing
                                cosine similarities in range [-1, 1].
        """
        # Normalize embeddings to unit vectors along the embedding dimension
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=-1)

        # Compute cosine similarity via dot product of normalized vectors
        # (B, Q, D) @ (B, D, C) -> (B, Q, C)
        interaction_matrix = torch.bmm(query_norm, doc_norm.transpose(1, 2))

        return interaction_matrix


class RBFKernelLayer(nn.Module):
    """
    Applies Radial Basis Function (RBF) kernels to an interaction matrix to compute
    soft-match features (histograms) for each query term.

    This layer implements the 'Kernel Pooling' mechanism.
    """

    def __init__(self, means, sigmas):
        """
        Args:
            means (list or torch.Tensor): The centers (mu) of the RBF kernels.
            sigmas (list or torch.Tensor): The widths (sigma) of the RBF kernels.
        """
        super(RBFKernelLayer, self).__init__()

        if len(means) != len(sigmas):
            raise ValueError("Length of means and sigmas must match.")

        # Register means and sigmas as buffers so they are saved with the model
        # but not updated by the optimizer (fixed kernels).
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))

    def forward(self, interaction_matrix):
        """
        Args:
            interaction_matrix: Tensor of shape (batch_size, q_len, c_len) containing
                                cosine similarities [-1, 1].

        Returns:
            log_pooled_features: Tensor of shape (batch_size, q_len, num_kernels).
                                 Represents the log of the sum of kernel activations
                                 across the candidate text for each query word.
        """
        # Dimensions:
        # interaction_matrix: [B, Q, C]
        # means, sigmas: [K]

        # Expand dimensions for broadcasting:
        # Matrix: [B, Q, C, 1]
        # Kernels: [1, 1, 1, K]
        mat = interaction_matrix.unsqueeze(-1)
        mus = self.means.view(1, 1, 1, -1)
        sigs = self.sigmas.view(1, 1, 1, -1)

        # Compute RBF activations: exp( - (x - mu)^2 / (2 * sigma^2) )
        # Shape: [B, Q, C, K]
        rbf = torch.exp(-torch.pow(mat - mus, 2) / (2 * torch.pow(sigs, 2)))

        # Sum over candidate dimension (C) to get soft match counts
        # Shape: [B, Q, K]
        pooled = torch.sum(rbf, dim=2)

        # Apply Logarithm (Log-Sum) with epsilon for numerical stability
        # This creates the "Soft-Match Histogram" vectors
        epsilon = 1e-10
        log_pooled = torch.log(torch.clamp(pooled, min=epsilon))

        return log_pooled


class DepthwiseSeparableConv1D(nn.Module):
    """
    A depthwise separable convolution block consisting of a depthwise convolution,
    a pointwise convolution, and an activation function.

    This is used in the Span Prediction Module to efficiently encode local context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        bias=True,
        activation=True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            padding (int): Zero-padding added to both sides of the input.
            bias (bool): If True, adds a learnable bias to the output.
            activation (bool): If True, applies ReLU activation after the convolutions.
        """
        super(DepthwiseSeparableConv1D, self).__init__()

        # Depthwise Convolution: Groups == in_channels
        # Each input channel is convolved with its own set of filters (1 per channel).
        # This captures spatial patterns within each feature independently.
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        # Pointwise Convolution: Kernel size = 1
        # Mixes the channels projected by the depthwise step to create new features.
        self.pointwise = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )

        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, length).
               Note: PyTorch Conv1d expects channels as the second dimension.

        Returns:
            out: Tensor of shape (batch_size, out_channels, length_out)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.activation:
            x = self.activation(x)

        return x
