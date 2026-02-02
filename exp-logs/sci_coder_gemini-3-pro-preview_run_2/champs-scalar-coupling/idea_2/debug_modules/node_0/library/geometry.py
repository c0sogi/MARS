import numpy as np
import torch
from library.config import Config


def get_neighbors(
    coords, cutoff=Config.CUTOFF_RADIUS, max_neighbors=Config.MAX_NEIGHBORS
):
    """
    Identifies pairs of atoms within a specified cutoff distance.

    Args:
        coords (np.ndarray): Array of shape (N, 3) containing atom coordinates.
        cutoff (float): Maximum distance to consider an edge.
        max_neighbors (int): Maximum number of neighbors per atom.

    Returns:
        np.ndarray: Edge indices of shape (2, E), where row 0 is source and row 1 is target.
    """
    num_atoms = coords.shape[0]

    # Compute distance matrix (N, N)
    # Using broadcasting: (N, 1, 3) - (1, N, 3)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))

    # Create mask for valid edges
    # Exclude self-loops (dist > 1e-6) and enforce cutoff
    mask = (dist_matrix <= cutoff) & (dist_matrix > 1e-6)

    # Get indices
    src_indices, dst_indices = np.where(mask)
    distances = dist_matrix[src_indices, dst_indices]

    # Filter by max_neighbors
    # We want to keep the closest neighbors for each source atom
    final_src = []
    final_dst = []

    # Iterate over each atom to enforce the neighbor count limit
    # Since N is small (QM9 max ~29), this loop is negligible in overhead
    for i in range(num_atoms):
        # Find edges where atom i is the source
        mask_i = src_indices == i
        if not np.any(mask_i):
            continue

        dst_i = dst_indices[mask_i]
        dist_i = distances[mask_i]

        if len(dst_i) > max_neighbors:
            # Get indices of the k smallest distances
            # argsort returns indices relative to the slice
            sorted_idx = np.argsort(dist_i)[:max_neighbors]
            dst_i = dst_i[sorted_idx]

        final_src.append(np.full(len(dst_i), i, dtype=int))
        final_dst.append(dst_i)

    if len(final_src) > 0:
        edge_index = np.vstack([np.concatenate(final_src), np.concatenate(final_dst)])
    else:
        edge_index = np.empty((2, 0), dtype=int)

    return edge_index


def get_triplets(edge_index, num_atoms):
    """
    Identifies triplets (k, j, i) for directional message passing.
    A triplet consists of an incoming edge k->j and an outgoing edge j->i, where k != i.

    Args:
        edge_index (np.ndarray): Edge indices of shape (2, E).
        num_atoms (int): Number of atoms in the molecule.

    Returns:
        tuple:
            - triplets (np.ndarray): Shape (3, T) containing [k, j, i] indices.
            - edge_indices_kj (np.ndarray): Shape (T,) indices of edge k->j in edge_index.
            - edge_indices_ji (np.ndarray): Shape (T,) indices of edge j->i in edge_index.
    """
    if edge_index.shape[1] == 0:
        return (
            np.empty((3, 0), dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    row, col = edge_index[0], edge_index[1]

    # We need to find pairs of edges (e1, e2) such that:
    # e1: k -> j (target of e1 is j)
    # e2: j -> i (source of e2 is j)
    # and k != i

    # Broadcasting approach to find matches
    # This creates an (E, E) boolean matrix.
    # For QM9, E is small enough (~500 max) that 500^2 = 250,000 bools is trivial.

    # Match target of incoming (col) with source of outgoing (row)
    # col[e1] == row[e2]
    match_mask = col[:, np.newaxis] == row[np.newaxis, :]

    # Ensure no backtracking (k != i)
    # row[e1] != col[e2]
    no_backtrack_mask = row[:, np.newaxis] != col[np.newaxis, :]

    valid_triplets = match_mask & no_backtrack_mask

    # Get indices of edges that form triplets
    e1_indices, e2_indices = np.where(valid_triplets)

    # Extract atom indices
    k = row[e1_indices]
    j = col[e1_indices]
    i = col[e2_indices]

    triplets = np.vstack([k, j, i])

    return triplets, e1_indices, e2_indices


def compute_distances(coords, edge_index):
    """
    Computes Euclidean distances for the given edges.

    Args:
        coords (np.ndarray): Atom coordinates (N, 3).
        edge_index (np.ndarray): Edge indices (2, E).

    Returns:
        np.ndarray: Distances of shape (E,).
    """
    if edge_index.shape[1] == 0:
        return np.array([])

    src, dst = edge_index[0], edge_index[1]

    vecs = coords[dst] - coords[src]
    distances = np.linalg.norm(vecs, axis=1)

    return distances


def compute_angles(coords, triplets):
    """
    Computes bond angles for triplets k -> j -> i.
    The angle is at atom j.

    Args:
        coords (np.ndarray): Atom coordinates (N, 3).
        triplets (np.ndarray): Triplet indices (3, T) [k, j, i].

    Returns:
        np.ndarray: Angles in radians of shape (T,).
    """
    if triplets.shape[1] == 0:
        return np.array([])

    k, j, i = triplets[0], triplets[1], triplets[2]

    # Vector j -> k
    v_jk = coords[k] - coords[j]
    # Vector j -> i
    v_ji = coords[i] - coords[j]

    # Compute norms
    n_jk = np.linalg.norm(v_jk, axis=1)
    n_ji = np.linalg.norm(v_ji, axis=1)

    # Avoid division by zero
    n_jk = np.maximum(n_jk, 1e-9)
    n_ji = np.maximum(n_ji, 1e-9)

    # Normalize vectors
    u_jk = v_jk / n_jk[:, np.newaxis]
    u_ji = v_ji / n_ji[:, np.newaxis]

    # Dot product
    dot_prod = np.sum(u_jk * u_ji, axis=1)

    # Clip for numerical stability of arccos
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    angles = np.arccos(dot_prod)

    return angles
