import numpy as np
from library.config import Config


def compute_adjacency(structure, seq_len):
    """
    Computes the adjacency matrix for an RNA sequence based on its secondary structure.
    The graph representation includes both backbone connections (linear chain) and
    hydrogen bonds (base pairs) defined by the structure string.

    Args:
        structure (str): Dot-bracket notation string of the RNA structure.
        seq_len (int): The length of the sequence.

    Returns:
        np.ndarray: A binary adjacency matrix of shape (seq_len, seq_len).
                    1.0 indicates a connection, 0.0 otherwise.
    """
    # Initialize adjacency matrix
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)

    # 1. Add Backbone Connections
    # Each nucleotide i is connected to i-1 and i+1
    # Create indices for the range [0, seq_len-2]
    indices = np.arange(seq_len - 1)
    adj[indices, indices + 1] = 1.0
    adj[indices + 1, indices] = 1.0

    # 2. Add Base Pair Connections
    # Parse the dot-bracket structure string
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Add undirected edge between paired bases i and j
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


def compute_rwpe(adjacency_matrix, steps=None):
    """
    Computes Random Walk Positional Encodings (RWPE) / Structural Fingerprints.
    This function calculates the diagonal elements of the random walk transition
    matrix raised to various powers (steps). These values represent the probability
    of a random walker returning to the starting node after k steps, encoding
    local structural topology and volume.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of shape (seq_len, seq_len).
        steps (list, optional): List of integers representing the number of steps
                                for the random walk (powers of the matrix).
                                If None, defaults to [1, 2, 4, 8, 16].

    Returns:
        np.ndarray: Feature matrix of shape (seq_len, len(steps)).
    """
    if steps is None:
        steps = [1, 2, 4, 8, 16]

    seq_len = adjacency_matrix.shape[0]

    # Calculate Degree Matrix D
    # D_ii = sum(A_ij) over j
    degree = np.sum(adjacency_matrix, axis=1)

    # Handle potential isolated nodes to avoid division by zero
    # (Though rare in RNA backbones, good for robustness)
    degree[degree == 0] = 1.0

    # Calculate Transition Matrix T = D^-1 * A
    # We divide each row i by degree[i]
    # Using broadcasting: (seq_len, seq_len) / (seq_len, 1)
    transition_matrix = adjacency_matrix / degree[:, None]

    rwpe_features = []

    # Compute T^k for each k in steps and extract the diagonal
    for k in steps:
        # Compute matrix power
        # For small matrices (L=107), np.linalg.matrix_power is efficient
        t_k = np.linalg.matrix_power(transition_matrix, k)

        # Extract diagonal elements (P_ii^k)
        diag = np.diag(t_k)
        rwpe_features.append(diag)

    # Stack features to create output matrix (seq_len, num_steps)
    rwpe_features = np.stack(rwpe_features, axis=1)

    return rwpe_features.astype(np.float32)


def compute_signed_distance(structure, seq_len):
    """
    Computes the signed distance to the paired base for each nucleotide.
    This feature is used for the Signed Sinusoidal Pairing Distance embedding.

    Args:
        structure (str): Dot-bracket notation string.
        seq_len (int): Length of the sequence.

    Returns:
        np.ndarray: Array of shape (seq_len,) containing the signed distance.
                    - If base i is paired with j, value is (j - i).
                    - If base i is unpaired, value is 0.
    """
    distances = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Calculate signed distance
                # For the opening bracket at j paired with i: distance is i - j (positive)
                # For the closing bracket at i paired with j: distance is j - i (negative)

                # Current position is i (closing), paired with j (opening)
                dist_i = j - i
                distances[i] = dist_i

                # Paired position is j (opening), paired with i (closing)
                dist_j = i - j
                distances[j] = dist_j

    return distances
