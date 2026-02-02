import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.linalg
from library.config import Config


def parse_structure(structure_str):
    """
    Parses a dot-bracket structure string into a dictionary of base pairs.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        dict: Mapping from index i to index j if base i is paired with j.
              Includes both i->j and j->i.
    """
    stack = []
    pairs = {}
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def compute_laplacian_pe(structure_str, seq_len, k=8):
    """
    Computes Laplacian Positional Encodings (LPE) for an RNA structure.

    Constructs a graph with backbone and base-pair connections, computes
    the normalized Laplacian, and extracts the first k non-trivial eigenvectors.

    Args:
        structure_str (str): Dot-bracket structure string.
        seq_len (int): Length of the sequence.
        k (int): Number of eigenvectors to keep (excluding the trivial constant vector).

    Returns:
        np.ndarray: Shape (seq_len, k) containing the LPE features.
    """
    # 1. Construct Adjacency Matrix
    # Nodes 0..seq_len-1
    # Edges: Backbone (i, i+1) and Pairs (i, j)
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Backbone connectivity
    for i in range(seq_len - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # Base pair connectivity
    pairs = parse_structure(structure_str)
    for i, j in pairs.items():
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Symmetry

    # 2. Compute Normalized Laplacian
    # L_norm = I - D^(-1/2) * A * D^(-1/2)
    degrees = np.sum(adj, axis=1)

    # Inverse square root of degrees (handle 0 degree safely, though unlikely in RNA backbone)
    d_inv_sqrt = np.power(degrees, -0.5, where=(degrees > 0))
    d_inv_sqrt[degrees == 0] = 0.0

    D_inv_sqrt_mat = np.diag(d_inv_sqrt)

    # L = I - D^-0.5 A D^-0.5
    identity = np.eye(seq_len, dtype=np.float32)
    L_norm = identity - D_inv_sqrt_mat @ adj @ D_inv_sqrt_mat

    # 3. Eigendecomposition
    # eigh is for symmetric/hermitian matrices, returns sorted eigenvalues
    eigenvalues, eigenvectors = scipy.linalg.eigh(L_norm)

    # 4. Select Eigenvectors
    # The smallest eigenvalue is close to 0 (trivial, constant vector for connected graph).
    # We want the next k smallest (low frequency) which capture global topology.
    # Indices: 0 is trivial, 1..k are the first k non-trivial.

    # Handle edge case where seq_len <= k
    if seq_len <= k:
        selected_eigenvectors = eigenvectors[:, 1:]  # Skip trivial
        pad_width = k - selected_eigenvectors.shape[1]
        if pad_width > 0:
            padding = np.zeros((seq_len, pad_width), dtype=np.float32)
            selected_eigenvectors = np.hstack([selected_eigenvectors, padding])
    else:
        selected_eigenvectors = eigenvectors[:, 1 : k + 1]

    return selected_eigenvectors.astype(np.float32)


def get_positional_encoding(positions, d_model):
    """
    Computes sinusoidal positional encodings for given positions (distances).
    Handles signed distances naturally for upstream/downstream distinction.

    Args:
        positions (torch.Tensor): Tensor of integer positions/distances. Shape (...,).
        d_model (int): Embedding dimension.

    Returns:
        torch.Tensor: Shape (..., d_model).
    """
    # Formula:
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    # Ensure positions is float for calculation, keep dimensions
    positions = positions.float().unsqueeze(-1)  # (..., 1)

    # Compute the division term: 10000^(2i/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device)
        * -(np.log(10000.0) / d_model)
    )

    # Initialize output tensor
    pe = torch.zeros(*positions.shape[:-1], d_model, device=positions.device)

    # Calculate arguments
    args = positions * div_term

    # Fill even and odd indices
    pe[..., 0::2] = torch.sin(args)
    pe[..., 1::2] = torch.cos(args)

    return pe


def mcrmse_loss(y_true, y_pred, mask=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).
    This is the evaluation metric.

    MCRMSE = Mean( RMSE(col_1), RMSE(col_2), ... )

    Args:
        y_true (torch.Tensor): Ground truth. Shape (Batch, Seq_Len, Targets).
        y_pred (torch.Tensor): Predictions. Shape (Batch, Seq_Len, Targets).
        mask (torch.Tensor, optional): Boolean mask indicating valid positions.
                                       Shape (Batch, Seq_Len) or (Batch, Seq_Len, 1).
                                       True indicates valid (scored), False indicates ignored.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    if mask is not None:
        if mask.dim() < y_true.dim():
            mask = mask.unsqueeze(-1)

        # Squared Error
        se = (y_true - y_pred) ** 2

        # Apply mask (zero out invalid positions)
        se = se * mask

        # Count valid items per column (sum over batch and sequence)
        count = mask.sum(dim=(0, 1))

        # Sum of squared errors per column
        mse_sum = se.sum(dim=(0, 1))

        # RMSE per column (avoid div by zero)
        rmse_per_col = torch.sqrt(mse_sum / (count + 1e-8))

        # Average over columns
        loss = torch.mean(rmse_per_col)

    else:
        # Standard calculation without mask
        mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))
        rmse = torch.sqrt(mse)
        loss = torch.mean(rmse)

    return loss


def masked_mse_loss(y_true, y_pred, mask=None):
    """
    Calculates the Masked Mean Squared Error (MSE).
    This is the training objective function.

    Args:
        y_true (torch.Tensor): Ground truth.
        y_pred (torch.Tensor): Predictions.
        mask (torch.Tensor, optional): Boolean mask.

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    if mask is not None:
        if mask.dim() < y_true.dim():
            mask = mask.unsqueeze(-1)

        sq_diff = (y_true - y_pred) ** 2
        sq_diff = sq_diff * mask

        # Mean over all valid elements
        loss = sq_diff.sum() / (mask.sum() + 1e-8)
    else:
        loss = torch.mean((y_true - y_pred) ** 2)

    return loss


def load_data(mode="train", debug_samples=None):
    """
    Loads data from the pre-generated parquet metadata files.

    Args:
        mode (str): "train", "val", or "test".
        debug_samples (int, optional): If set, returns only the first N samples.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if mode == "train":
        path = Config.TRAIN_PATH
    elif mode == "val":
        path = Config.VAL_PATH
    elif mode == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at {path}. Ensure metadata is generated."
        )

    df = pd.read_parquet(path)

    if debug_samples is not None:
        df = df.iloc[:debug_samples]

    return df
