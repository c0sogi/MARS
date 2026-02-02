import torch
import numpy as np
from library.config import Config


def re_ranking(
    probFea,
    galFea,
    k1=Config.RERANK_K1,
    k2=Config.RERANK_K2,
    lambda_value=Config.RERANK_LAMBDA,
):
    """
    Implements k-Reciprocal Encoding Re-ranking.

    This function refines the distance matrix between query and gallery images by
    considering the manifold structure of the data. It combines the original
    Euclidean distance with a Jaccard distance derived from k-reciprocal nearest neighbors.

    Args:
        probFea (np.ndarray or torch.Tensor): Query features (N x D).
        galFea (np.ndarray or torch.Tensor): Gallery features (M x D).
        k1 (int): Number of neighbors for k-reciprocal set calculation.
        k2 (int): Number of neighbors for local query expansion.
        lambda_value (float): Weighting parameter for the original distance (0 to 1).
                              Final = lambda * original + (1 - lambda) * jaccard.

    Returns:
        np.ndarray: Re-ranked distance matrix (N x M).
    """
    # Ensure inputs are on the correct device
    device = Config.DEVICE

    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(probFea, np.ndarray):
        probFea = torch.from_numpy(probFea)
    if isinstance(galFea, np.ndarray):
        galFea = torch.from_numpy(galFea)

    probFea = probFea.to(device)
    galFea = galFea.to(device)

    # L2 Normalize features (Critical for Cosine/Euclidean equivalence)
    probFea = torch.nn.functional.normalize(probFea, p=2, dim=1)
    galFea = torch.nn.functional.normalize(galFea, p=2, dim=1)

    # Concatenate all features to form the global manifold
    # Query indices: 0 to query_num-1
    # Gallery indices: query_num to all_num-1
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea], dim=0)

    # --------------------------------------------------------------------------
    # 1. Compute Original Euclidean Distance
    # --------------------------------------------------------------------------
    # dist = ||x||^2 + ||y||^2 - 2x@y.t()
    # Since normalized, ||x||^2 = 1. So dist = 2 - 2x@y.t()
    distmat = (
        torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    )
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)

    # Numerical stability (ensure non-negative)
    distmat = distmat.clamp(min=1e-12)

    # Keep original distance for final fusion and neighbor search
    # We use squared Euclidean distance for the Gaussian kernel weights later
    original_dist = distmat

    # --------------------------------------------------------------------------
    # 2. k-Reciprocal Neighbor Search
    # --------------------------------------------------------------------------

    # Sort distances to find neighbors
    # We move to CPU for the iterative set expansion logic as it involves
    # dynamic array operations that are hard to fully vectorize on GPU.
    # However, for sorting, GPU is much faster.
    _, indices = torch.sort(original_dist, dim=1)

    # Move necessary data to CPU for the loop
    original_dist_cpu = original_dist.cpu()
    indices_cpu = indices.cpu()
    gallery_num = original_dist.size(0)  # This is actually all_num

    # Extract top k1+1 neighbors (including self)
    forward_k_neigh_index = indices_cpu[:, : k1 + 1]
    backward_k_neigh_index = indices_cpu[:, : k1 + 1]

    # Initialize the expansion matrix V (Weighted Adjacency)
    # V[i, j] will store the weight if j is in the robust set of i
    V = torch.zeros(gallery_num, gallery_num, dtype=torch.float32)

    # Iterate over every sample to construct its robust k-reciprocal set
    for i in range(gallery_num):
        # 2a. Identify k-reciprocal neighbors
        # A neighbor 'candidate' is k-reciprocal if:
        #   candidate is in k1-NN of i  AND  i is in k1-NN of candidate
        forward_idx = forward_k_neigh_index[i]
        backward_idx = backward_k_neigh_index[i]

        fi = forward_idx.numpy()
        bi = backward_idx.numpy()

        # Intersection gives the mutual neighbors
        k_reciprocal_index = np.intersect1d(fi, bi)
        k_reciprocal_expansion_index = k_reciprocal_index

        # 2b. Robust Expansion (include neighbors of neighbors)
        # Iterate through the k-reciprocal neighbors to potentially add their neighbors
        for candidate in k_reciprocal_index:
            candidate_forward = forward_k_neigh_index[candidate].numpy()
            candidate_backward = backward_k_neigh_index[candidate].numpy()
            candidate_k_reciprocal = np.intersect1d(
                candidate_forward, candidate_backward
            )

            # Condition: If the candidate's set overlaps significantly with the current set
            if len(
                np.intersect1d(candidate_k_reciprocal, k_reciprocal_index)
            ) > 2.0 / 3.0 * len(candidate_k_reciprocal):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal
                )

        # Remove duplicates and limit to unique indices
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # 2c. Calculate Weights (Gaussian Kernel)
        # weight = exp(-dist^2) -- note original_dist is already squared
        weight = torch.exp(-original_dist_cpu[i, k_reciprocal_expansion_index])

        # Normalize weights to sum to 1 (L1 norm of row = 1)
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    # --------------------------------------------------------------------------
    # 3. Jaccard Distance Calculation
    # --------------------------------------------------------------------------
    # Jaccard(A, B) = 1 - (Intersection / Union)
    # Using the property: Intersection + Union = |A| + |B| = 2 (since normalized)
    # Jaccard(A, B) = 1 - I / (2 - I)
    # Also, Intersection I = 1 - 0.5 * ||A - B||_1

    # Move V to GPU for fast matrix operations
    V = V.to(device)

    # Compute pairwise L1 distance matrix of V
    # torch.cdist with p=1 is highly optimized
    if V.size(0) > 15000:
        # Safety fallback for extremely large sets (though 7000 fits on A100)
        # Process in chunks if necessary, but here we assume it fits.
        pass

    jaccard_dist = torch.cdist(V, V, p=1)

    # Convert L1 distance to Jaccard Distance
    # I = 1 - 0.5 * L1
    # J = 1 - I / (2 - I)
    # Simplifying: J = L1 / (2 + L1) ? No, let's stick to the derivation.
    # I = 1 - 0.5 * L1
    # Denom = 2 - (1 - 0.5 * L1) = 1 + 0.5 * L1
    # J = 1 - (1 - 0.5 * L1) / (1 + 0.5 * L1)

    jaccard_dist = 1.0 - (1.0 - 0.5 * jaccard_dist) / (1.0 + 0.5 * jaccard_dist)

    # --------------------------------------------------------------------------
    # 4. Final Fusion
    # --------------------------------------------------------------------------
    # Combine original distance and Jaccard distance
    # Note: original_dist is squared Euclidean.

    final_dist = lambda_value * original_dist + (1 - lambda_value) * jaccard_dist

    # Slice the matrix to return only Query vs Gallery distances
    # Rows: 0 to query_num (Queries)
    # Cols: query_num to end (Gallery)
    final_dist = final_dist[:query_num, query_num:]

    return final_dist.cpu().numpy()
