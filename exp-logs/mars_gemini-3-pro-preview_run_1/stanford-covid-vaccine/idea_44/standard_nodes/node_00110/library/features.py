import numpy as np
import torch
from library.config import Config

# =========================================================================
# Tokenization Dictionaries
# =========================================================================

# Atomic Sequence Tokens
TOKEN_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}

# Predicted Loop Type Tokens
# S: Stem, M: Multiloop, I: Internal loop, B: Bulge, H: Hairpin, E: Dangling End, X: External loop
TOKEN_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Structure Processing Functions
# =========================================================================


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: An array of shape (len(structure),) where arr[i] is the index
                    of the base paired with i. Returns -1 if i is unpaired.
    """
    seq_len = len(structure)
    pairs = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
            else:
                # Unbalanced closing parenthesis, technically invalid structure
                # but we handle gracefully by leaving it as -1
                pass

    return pairs


def get_signed_distance_vector(structure):
    """
    Calculates the signed distance between paired bases.

    Distance = partner_index - current_index.
    - Upstream dependencies (paired with later base) have Positive distance.
    - Downstream dependencies (paired with earlier base) have Negative distance.
    - Unpaired bases have 0 distance.

    Args:
        structure (str): Dot-bracket notation string.

    Returns:
        np.ndarray: Array of shape (len(structure),) containing signed distances.
    """
    pairs = get_couples(structure)
    indices = np.arange(len(structure))

    # Calculate distance: partner - current
    # If pairs[i] is -1 (unpaired), result would be -1 - i. We need to mask this.
    distances = pairs - indices

    # Set unpaired positions to 0
    # A distance of 0 is distinct because a base cannot pair with itself (dist=0 implies i==j)
    distances[pairs == -1] = 0

    return distances


# =========================================================================
# Encoding Functions
# =========================================================================


def sinusoidal_encoding(values, dim):
    """
    Generates fixed sinusoidal embeddings for input values (distances).

    Uses the formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Handles negative values naturally via odd/even properties of sin/cos.

    Args:
        values (np.ndarray): Input values (e.g., signed distances) of shape (seq_len,).
        dim (int): The embedding dimension. Must be even.

    Returns:
        np.ndarray: Embeddings of shape (seq_len, dim).
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    seq_len = len(values)

    # Create frequency term: 10000^(2i/dim)
    # shape: (dim/2,)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

    # Compute arguments: pos * div_term
    # values shape: (seq_len, 1)
    # div_term shape: (1, dim/2)
    # args shape: (seq_len, dim/2)
    args = values[:, np.newaxis] * div_term[np.newaxis, :]

    # Create embedding matrix
    pe = np.zeros((seq_len, dim), dtype=np.float32)

    # Apply sin to even indices
    pe[:, 0::2] = np.sin(args)

    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(args)

    return pe


# =========================================================================
# Tokenization Helpers
# =========================================================================


def tokenize_sequence(sequence):
    """
    Converts an RNA sequence string to an integer array.

    Args:
        sequence (str): RNA sequence (A, G, C, U).

    Returns:
        np.ndarray: Array of token indices.
    """
    return np.array([TOKEN_SEQ.get(base, 0) for base in sequence], dtype=np.int64)


def tokenize_loop(loop_type):
    """
    Converts a predicted loop type string to an integer array.

    Args:
        loop_type (str): Loop type string (S, M, I, B, H, E, X).

    Returns:
        np.ndarray: Array of token indices.
    """
    return np.array([TOKEN_LOOP.get(lt, 0) for lt in loop_type], dtype=np.int64)
