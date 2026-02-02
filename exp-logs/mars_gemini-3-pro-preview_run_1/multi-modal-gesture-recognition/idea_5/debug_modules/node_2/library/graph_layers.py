import torch
import torch.nn as nn
import numpy as np
import math
from library.config import SKELETON_EDGES, NUM_JOINTS


def get_adjacency_matrix(num_joints=NUM_JOINTS, edges=SKELETON_EDGES):
    """
    Constructs the binary physical adjacency matrix for the skeleton.

    Args:
        num_joints (int): Number of joints in the skeleton.
        edges (list of tuples): List of connected joint pairs (indices).

    Returns:
        np.ndarray: A (num_joints, num_joints) binary adjacency matrix
                    with self-loops and symmetric connections.
    """
    adj = np.zeros((num_joints, num_joints), dtype=np.float32)

    # Add self-loops
    for i in range(num_joints):
        adj[i, i] = 1.0

    # Add edges (undirected)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


class AdaptiveGraphConv(nn.Module):
    """
    Adaptive Spatial Graph Convolution Layer.

    Implements the operation: Y = ReLU((A_phys + A_learn) X W)

    Where:
        A_phys: Fixed physical adjacency matrix (with self-loops).
        A_learn: Learnable adjacency matrix (initialized to 0).
        X: Input features.
        W: Learnable weight matrix.
    """

    def __init__(
        self, in_channels, out_channels, num_joints=NUM_JOINTS, adj_matrix=None
    ):
        """
        Args:
            in_channels (int): Number of input features per joint.
            out_channels (int): Number of output features per joint.
            num_joints (int): Number of joints in the graph.
            adj_matrix (np.ndarray, optional): Pre-computed adjacency matrix.
                                               If None, computed from config.
        """
        super(AdaptiveGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints

        # Initialize Adjacency Matrix
        if adj_matrix is None:
            adj_matrix = get_adjacency_matrix(num_joints, SKELETON_EDGES)

        # Register A_phys as a buffer (not a parameter, but part of state_dict)
        # Shape: (V, V)
        self.register_buffer("A_phys", torch.from_numpy(adj_matrix).float())

        # A_learn is a learnable parameter, initialized to 0
        # Shape: (V, V)
        self.A_learn = nn.Parameter(torch.zeros(num_joints, num_joints))

        # Weight matrix W
        # Shape: (in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

        self.relu = nn.ReLU()

    def reset_parameters(self):
        """
        Initialize parameters using Kaiming uniform initialization for weights
        and uniform initialization for bias.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # A_learn is already initialized to zeros in __init__

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Joints, In_Channels)
                              or (Batch, Time, Joints * In_Channels) if flattened,
                              but strictly expects (B, T, V, C) for logic.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Time, Joints, Out_Channels)
        """
        # Ensure input is 4D: (B, T, V, C)
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (Batch, Time, Joints, Channels), got {x.shape}"
            )

        B, T, V, C = x.shape

        if V != self.num_joints:
            raise ValueError(f"Input has {V} joints, expected {self.num_joints}")
        if C != self.in_channels:
            raise ValueError(f"Input has {C} channels, expected {self.in_channels}")

        # Combine Batch and Time dimensions for matrix multiplication
        # x_reshaped: (B*T, V, C)
        x_reshaped = x.view(B * T, V, C)

        # Construct Adaptive Adjacency Matrix
        # A = A_phys + A_learn
        # A shape: (V, V)
        A = self.A_phys + self.A_learn

        # 1. Graph Convolution (Aggregate neighbors)
        # Z = A * X
        # (V, V) @ (B*T, V, C) -> (B*T, V, C)
        # We need to broadcast A over B*T.
        # torch.matmul handles (V,V) x (V,C) -> (V,C) if we iterate,
        # or we can treat x as a batch of matrices.
        # A is (V, V), x_reshaped is (N, V, C).
        # We want A applied to every (V, C) matrix in the batch.
        # torch.matmul(A, x) broadcasts if A is (V, V) and x is (N, V, C) -> result (N, V, C)
        # Wait, torch.matmul(A, x) where A=(V,V) and x=(N,V,C) usually requires A to be (N,V,V) or broadcasting.
        # Actually, torch.matmul supports broadcasting. If arg1 is (n, m) and arg2 is (B, m, p), result is (B, n, p).
        # Let's verify: (V, V) @ (B*T, V, C) -> (B*T, V, C). Yes, this works in PyTorch.

        z = torch.matmul(A, x_reshaped)

        # 2. Feature Transformation (Linear projection)
        # Y = Z * W
        # (B*T, V, C_in) @ (C_in, C_out) -> (B*T, V, C_out)
        y = torch.matmul(z, self.weight)

        # 3. Bias and Activation
        y = y + self.bias
        y = self.relu(y)

        # Reshape back to (B, T, V, C_out)
        y = y.view(B, T, V, self.out_channels)

        return y
