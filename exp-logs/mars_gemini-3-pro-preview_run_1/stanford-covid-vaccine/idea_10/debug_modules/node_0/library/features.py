import numpy as np


def parse_structure(structure_str):
    """
    Parses a dot-bracket structure string to identify base pairs.

    This function iterates through the structure string, using a stack to track
    opening parentheses. When a closing parenthesis is found, it pops the
    corresponding opening index to form a pair.

    Args:
        structure_str (str): The RNA secondary structure in dot-bracket notation
                             (e.g., "((..))").

    Returns:
        list of tuple: A list of (start_index, end_index) tuples representing
                       base pairs, where start_index < end_index.
    """
    stack = []
    pairs = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs.append((start, i))
    return pairs


def get_distance_encoding(structure_str, seq_len):
    """
    Generates a scalar distance encoding for sinusoidal embedding based on base pairs.

    For a paired base at index i paired with j, the distance is calculated as j - i.
    - If i < j (opening base), the distance is positive.
    - If i > j (closing base), the distance is negative.
    - Unpaired bases ('.') are assigned a distance of 0.

    This explicit geometric encoding helps the model understand the spatial
    relationships defined by the secondary structure.

    Args:
        structure_str (str): The RNA secondary structure in dot-bracket notation.
        seq_len (int): The total length of the sequence (e.g., 107).

    Returns:
        np.ndarray: A float32 array of shape (seq_len,) containing the distances.
    """
    pairs = parse_structure(structure_str)
    distances = np.zeros(seq_len, dtype=np.float32)

    for start, end in pairs:
        # Calculate distance
        dist = end - start

        # Assign positive distance to opening base
        distances[start] = dist

        # Assign negative distance to closing base
        distances[end] = -dist

    return distances


def get_paired_index_map(structure_str, seq_len):
    """
    Creates an index mapping for the Paired-Base Identity feature.

    This feature allows the model to "teleport" semantic information (nucleotide identity)
    from a paired base to the current position.
    - For a base at index i paired with base j, the value at index i is j.
    - For unpaired bases, the value is -1 (which will be handled/masked by the model).

    Args:
        structure_str (str): The RNA secondary structure in dot-bracket notation.
        seq_len (int): The total length of the sequence (e.g., 107).

    Returns:
        np.ndarray: An int32 array of shape (seq_len,) containing paired indices.
    """
    pairs = parse_structure(structure_str)
    pair_indices = np.full(seq_len, -1, dtype=np.int32)

    for start, end in pairs:
        # Map start to end
        pair_indices[start] = end

        # Map end to start
        pair_indices[end] = start

    return pair_indices
