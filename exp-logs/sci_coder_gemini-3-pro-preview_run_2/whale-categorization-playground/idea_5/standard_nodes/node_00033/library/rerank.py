import numpy as np
import torch


def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    K-Reciprocal Encoding Re-ranking.

    This function re-ranks the distances between query and gallery features by
    considering the k-reciprocal nearest neighbors, which captures the manifold
    structure of the data.

    Reference:
    Zhong Z, Zheng L, Cao D, Li S. Re-ranking Person Re-identification with
    k-reciprocal Encoding. CVPR 2017.

    Args:
        probFea (numpy.ndarray or torch.Tensor): Query features. Shape (N, D).
        galFea (numpy.ndarray or torch.Tensor): Gallery features. Shape (M, D).
        k1 (int): The number of neighbors to consider for k-reciprocal search.
                  Defaults to 20.
        k2 (int): The number of neighbors to consider for local query expansion.
                  Defaults to 6.
        lambda_value (float): Weighting factor for combining Jaccard distance
                              with original distance. Defaults to 0.3.

    Returns:
        numpy.ndarray: The re-ranked distance matrix of shape (N, M).
    """
    # ---------------------------------------------------------
    # 1. Input Standardization
    # ---------------------------------------------------------
    if isinstance(probFea, torch.Tensor):
        probFea = probFea.cpu().numpy()
    if isinstance(galFea, torch.Tensor):
        galFea = galFea.cpu().numpy()

    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]

    # Concatenate query and gallery features
    feat = np.concatenate([probFea, galFea], axis=0)

    # ---------------------------------------------------------
    # 2. Initial Distance Calculation
    # ---------------------------------------------------------
    # L2 Normalize features
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / (norm + 1e-8)

    # Compute Pairwise Euclidean Distance
    # dist = ||x||^2 + ||y||^2 - 2x.y
    # Since normalized, ||x||^2 = 1, so dist = 2 - 2x.y
    dist = 2.0 - 2.0 * np.dot(feat, feat.T)

    # Numerical stability (fix negative zeros)
    dist = np.maximum(dist, 0.0)

    # Keep original distance for final fusion
    original_dist = dist.copy()

    # Get initial ranking (indices of sorted distances)
    # We sort the full matrix to easily access neighbors
    initial_rank = np.argsort(dist, axis=1)

    # ---------------------------------------------------------
    # 3. k-Reciprocal Neighbor Calculation
    # ---------------------------------------------------------
    # V matrix stores the weighted neighbor information for Jaccard calculation
    V = np.zeros((all_num, all_num), dtype=np.float32)

    # Iterate over every sample (query + gallery)
    for i in range(all_num):
        # Forward neighbors: The k1 closest to i
        forward_k_neigh_index = initial_rank[i, : k1 + 1]

        # Backward neighbors: For each forward neighbor, who are their k1 closest?
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]

        # k-Reciprocal: Neighbors who also have 'i' in their top k1
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index

        # Local Query Expansion (LQE)
        # Iterate through reciprocal neighbors to find their robust neighbors
        for candidate in k_reciprocal_index:
            candidate_forward_k_neigh_index = initial_rank[candidate, : k1 // 2 + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : k1 // 2 + 1
            ]

            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            # Condition: Significant overlap between candidate's set and current set
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # Calculate weights based on original distance
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    # ---------------------------------------------------------
    # 4. Jaccard Distance & Fusion
    # ---------------------------------------------------------
    # We only compute Jaccard distance for the Query rows to save time
    # Jaccard(A, B) = 1 - (Intersection / Union)
    # Intersection = sum(min(Va, Vb))
    # Union = sum(max(Va, Vb))

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        # Vectorized comparison of Query 'i' against all samples
        # V[i] shape: (all_num,)
        # V shape: (all_num, all_num)

        temp_min = np.minimum(V[i], V)
        intersection = np.sum(temp_min, axis=1)

        temp_max = np.maximum(V[i], V)
        union = np.sum(temp_max, axis=1)

        jaccard_dist[i] = 1.0 - intersection / (union + 1e-8)

    # ---------------------------------------------------------
    # 5. Final Output
    # ---------------------------------------------------------
    # Combine Original and Jaccard distances
    final_dist = jaccard_dist * lambda_value + original_dist[:query_num, :] * (
        1 - lambda_value
    )

    # Return only the Query-to-Gallery block
    # The gallery indices in the concatenated matrix start at 'query_num'
    return final_dist[:, query_num:]
