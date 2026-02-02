import torch
import numpy as np
from library.config import Config


def compute_dist(pos, edge_index):
    """
    Computes Euclidean distances and direction vectors for edges.

    Args:
        pos (torch.Tensor): Node positions (N, 3).
        edge_index (torch.Tensor): Edge indices (2, E), where row 0 is source, row 1 is target.

    Returns:
        dist (torch.Tensor): Euclidean distances (E,).
        vec (torch.Tensor): Direction vectors (E, 3) pointing from source to target (r_i - r_j).
    """
    j, i = edge_index

    # Vector from source (j) to target (i)
    vec = pos[i] - pos[j]

    # Euclidean distance
    dist = torch.norm(vec, p=2, dim=-1)

    return dist, vec


def get_triplets(edge_index, num_nodes):
    """
    Identifies triplets k -> j -> i for directional message passing.

    Args:
        edge_index (torch.Tensor): Edge indices (2, E).
        num_nodes (int): Number of nodes (atoms) in the molecule.

    Returns:
        triplet_indices (torch.Tensor): Indices of edge pairs (2, T).
                                        Row 0: index of incoming edge (k->j).
                                        Row 1: index of outgoing edge (j->i).
    """
    src = edge_index[0]
    dst = edge_index[1]

    # We want to find pairs of edges (e_in, e_out) such that:
    # 1. target(e_in) == source(e_out)  (Connectivity at node j)
    # 2. source(e_in) != target(e_out)  (No backtracking k != i)

    # Broadcasting to find connections: (E, 1) == (1, E) -> (E, E) matrix
    # Rows correspond to e_in, Columns correspond to e_out
    # adjacency_match[m, n] is True if target of edge m == source of edge n
    adjacency_match = dst.unsqueeze(1) == src.unsqueeze(0)

    # Broadcasting to filter backtracking: (E, 1) != (1, E)
    # non_backtrack[m, n] is True if source of edge m != target of edge n
    non_backtrack = src.unsqueeze(1) != dst.unsqueeze(0)

    # Combine masks
    valid_triplets = adjacency_match & non_backtrack

    # Get indices of True values
    # nonzero() returns (T, 2), we transpose to (2, T)
    # Row 0 will be indices of incoming edges, Row 1 indices of outgoing edges
    triplet_indices = valid_triplets.nonzero().t()

    return triplet_indices


def compute_angles(pos, edge_index, triplet_indices):
    """
    Computes the angle theta_ijk at node j for triplets k->j->i.

    Args:
        pos (torch.Tensor): Node positions (N, 3).
        edge_index (torch.Tensor): Edge indices (2, E).
        triplet_indices (torch.Tensor): Indices of edge pairs (2, T).

    Returns:
        angles (torch.Tensor): Angles in radians (T,).
    """
    # Extract edge indices for the triplets
    idx_kj = triplet_indices[0]  # Incoming edges k->j
    idx_ji = triplet_indices[1]  # Outgoing edges j->i

    # Get atom indices
    # Edge k->j
    k = edge_index[0, idx_kj]
    j = edge_index[1, idx_kj]
    # Edge j->i
    i = edge_index[1, idx_ji]

    # Calculate vectors radiating from the central atom j
    # r_jk = pos[k] - pos[j]
    # r_ji = pos[i] - pos[j]
    r_jk = pos[k] - pos[j]
    r_ji = pos[i] - pos[j]

    # Normalize vectors
    n_jk = torch.nn.functional.normalize(r_jk, p=2, dim=-1)
    n_ji = torch.nn.functional.normalize(r_ji, p=2, dim=-1)

    # Compute Dot Product: cos(theta) = (a . b) / (|a| |b|) (already normalized)
    dot = (n_jk * n_ji).sum(dim=-1)

    # Clamp values to [-1, 1] to avoid numerical errors with acos
    dot = torch.clamp(dot, -0.95, 0.95)

    # Compute Angle
    angles = torch.acos(dot)

    return angles


def compute_dist_and_angle(pos, edge_index, triplet_indices):
    """
    Wrapper to compute both distances and angles.

    Args:
        pos (torch.Tensor): Node positions (N, 3).
        edge_index (torch.Tensor): Edge indices (2, E).
        triplet_indices (torch.Tensor): Triplet indices (2, T).

    Returns:
        dist (torch.Tensor): Distances (E,).
        angles (torch.Tensor): Angles (T,).
    """
    dist, _ = compute_dist(pos, edge_index)
    angles = compute_angles(pos, edge_index, triplet_indices)
    return dist, angles


def process_molecule(pos, atom_types=None, cutoff=Config.CUTOFF):
    """
    Full pipeline to process a single molecule's geometry.
    Generates the radius graph, computes distances, finds triplets, and computes angles.

    Args:
        pos (torch.Tensor or np.ndarray): Node positions (N, 3).
        atom_types (optional): Not used for geometry but kept for API consistency.
        cutoff (float): Radius cutoff for graph construction.

    Returns:
        dict: containing 'edge_index', 'dist', 'vec', 'triplet_indices', 'angles'
    """
    # Ensure input is tensor
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=torch.float32)

    num_nodes = pos.shape[0]

    # 1. Generate Radius Graph
    # Compute pairwise distance matrix
    dist_mat = torch.cdist(pos, pos)

    # Create adjacency mask (dist < cutoff AND dist > 0 to remove self-loops)
    # 1e-4 tolerance for self-loop check
    mask = (dist_mat < cutoff) & (dist_mat > 1e-4)

    # Get edge indices (2, E)
    edge_index = mask.nonzero().t()

    # Handle case with no edges (e.g. single atom or disconnected)
    if edge_index.numel() == 0:
        return {
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "dist": torch.empty((0,), dtype=torch.float32),
            "vec": torch.empty((0, 3), dtype=torch.float32),
            "triplet_indices": torch.empty((2, 0), dtype=torch.long),
            "angles": torch.empty((0,), dtype=torch.float32),
        }

    # 2. Compute Edge Features
    dist, vec = compute_dist(pos, edge_index)

    # 3. Compute Triplets
    triplet_indices = get_triplets(edge_index, num_nodes)

    # 4. Compute Angles
    if triplet_indices.numel() > 0:
        angles = compute_angles(pos, edge_index, triplet_indices)
    else:
        angles = torch.empty((0,), dtype=torch.float32)

    return {
        "edge_index": edge_index,
        "dist": dist,
        "vec": vec,
        "triplet_indices": triplet_indices,
        "angles": angles,
    }
