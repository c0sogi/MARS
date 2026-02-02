import numpy as np
import torch
import torch.nn.functional as F
from library.config import CFG


def re_ranking(probFea, galFea, k1, k2, lambda_value):
    """
    Implements k-Reciprocal Encoding Re-ranking.

    This function computes a re-ranked distance matrix between query (probe) features
    and gallery features. It combines the original Euclidean distance with a Jaccard
    distance derived from k-reciprocal nearest neighbors, improving robustness
    for open-set identification.

    Args:
        probFea (torch.Tensor or np.ndarray): Query features with shape (M, D).
        galFea (torch.Tensor or np.ndarray): Gallery features with shape (N, D).
        k1 (int): The number of neighbors to consider for the k-reciprocal set.
        k2 (int): The threshold for query expansion.
        lambda_value (float): The weighting parameter between the Jaccard distance
                              and the original distance (0 <= lambda_value <= 1).

    Returns:
        np.ndarray: The re-ranked distance matrix with shape (M, N).
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure inputs are tensors and on the correct device
    if isinstance(probFea, np.ndarray):
        probFea = torch.from_numpy(probFea)
    if isinstance(galFea, np.ndarray):
        galFea = torch.from_numpy(galFea)

    probFea = probFea.to(device).float()
    galFea = galFea.to(device).float()

    # L2 Normalize features
    # This ensures Euclidean distance is equivalent to Cosine distance
    probFea = F.normalize(probFea, p=2, dim=1)
    galFea = F.normalize(galFea, p=2, dim=1)

    # Dimensions
    num_query = probFea.size(0)
    num_gallery = galFea.size(0)
    num_all = num_query + num_gallery

    # Concatenate features to form the complete set for neighbor search
    features = torch.cat([probFea, galFea], dim=0)

    # -------------------------------------------------------------------------
    # 1. Compute Original Distance Matrix (Euclidean)
    # -------------------------------------------------------------------------
    # dist(x, y) = ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    # Since features are normalized, ||x||^2 = 1, so dist = 2 - 2<x, y>
    distmat = 2.0 - 2.0 * torch.mm(features, features.t())
    distmat = distmat.clamp(min=0.0).sqrt()

    # -------------------------------------------------------------------------
    # 2. k-Reciprocal Neighbor Search
    # -------------------------------------------------------------------------
    # We perform the complex set operations on CPU using Numpy
    original_dist = distmat.cpu().numpy()

    # Get the top (k1 + 1) neighbors for every sample
    # largest=False gives smallest distances
    _, initial_rank = torch.topk(distmat, k=k1 + 1, dim=1, largest=False)
    initial_rank = initial_rank.cpu().numpy()

    # Initialize the weighted neighbor matrix V
    V = np.zeros((num_all, num_all), dtype=np.float32)

    # Iterate over all samples to find k-reciprocal sets
    for i in range(num_all):
        # Forward neighbors: N(p, k1)
        forward_k_neigh_index = initial_rank[i, : k1 + 1]

        # Backward neighbors: N(g, k1) for each g in N(p, k1)
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]

        # k-Reciprocal condition: g in N(p, k1) AND p in N(g, k1)
        # We check where 'i' appears in the backward neighbor lists
        fi = np.where(backward_k_neigh_index == i)[0]

        # Indices in forward_k_neigh_index that satisfy reciprocity
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index

        # Query Expansion
        # If the reciprocal set is robust (size > k2), expand it with neighbors of neighbors
        if len(k_reciprocal_index) > k2:
            for candidate in k_reciprocal_index:
                # Add 1/2 * k1 neighbors of the candidate
                candidate_neigh_index = initial_rank[
                    candidate, : int(np.round(k1 / 2)) + 1
                ]
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_neigh_index
                )

        # Unique neighbors only
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # Weight Calculation
        # Weight is based on the exponential of the negative original distance
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])

        # Assign weights to the V matrix
        V[i, k_reciprocal_expansion_index] = weight

    # -------------------------------------------------------------------------
    # 3. Compute Jaccard Distance
    # -------------------------------------------------------------------------
    # Move V back to GPU for fast matrix operations
    V_tensor = torch.from_numpy(V).to(device)

    # Soft Jaccard / Tanimoto Distance
    # J(A, B) = 1 - (A . B) / (||A||^2 + ||B||^2 - A . B)

    # Dot product (intersection proxy)
    ab = torch.mm(V_tensor, V_tensor.t())

    # Squared norms (union proxy components)
    aa = V_tensor.pow(2).sum(dim=1, keepdim=True).expand(num_all, num_all)
    bb = aa.t()

    # Compute Jaccard Distance
    jaccard_dist = 1.0 - ab / (aa + bb - ab + 1e-12)

    # -------------------------------------------------------------------------
    # 4. Final Combination
    # -------------------------------------------------------------------------
    # Combine Jaccard distance with original distance
    # original_dist is currently numpy, use the tensor version 'distmat'
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * distmat

    # Extract the sub-matrix corresponding to Query (rows) vs Gallery (cols)
    # Rows: 0 to num_query
    # Cols: num_query to end
    final_dist = final_dist[:num_query, num_query:]

    return final_dist.cpu().numpy()
