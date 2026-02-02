import torch
import numpy as np
import gc
from library.config import Config


def re_ranking(
    probFea,
    galFea,
    k1=Config.RERANK_K1,
    k2=Config.RERANK_K2,
    lambda_value=Config.RERANK_LAMBDA,
):
    """
    Implements k-Reciprocal Encoding Re-ranking (Zhong et al., CVPR 2017).

    Args:
        probFea (torch.Tensor or np.ndarray): Query features (N x D).
        galFea (torch.Tensor or np.ndarray): Gallery features (M x D).
        k1 (int): The number of neighbors to consider for reciprocity.
        k2 (int): The number of neighbors to consider for local query expansion.
        lambda_value (float): Weighting parameter for the original distance.

    Returns:
        np.ndarray: Re-ranked distance matrix (N x M).
    """

    # Ensure inputs are Torch tensors on the correct device
    device = Config.DEVICE

    if isinstance(probFea, np.ndarray):
        probFea = torch.from_numpy(probFea)
    if isinstance(galFea, np.ndarray):
        galFea = torch.from_numpy(galFea)

    probFea = probFea.to(device)
    galFea = galFea.to(device)

    # Normalize features (L2 normalization)
    # This ensures Euclidean distance is proportional to Cosine distance
    probFea = torch.nn.functional.normalize(probFea, p=2, dim=1)
    galFea = torch.nn.functional.normalize(galFea, p=2, dim=1)

    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)

    # Concatenate to form the full set (Query + Gallery)
    # This is necessary to find reciprocal neighbors within the gallery itself
    feat = torch.cat([probFea, galFea], dim=0)

    # -------------------------------------------------------------------------
    # 1. Compute Pairwise Euclidean Distance
    # -------------------------------------------------------------------------
    # dist(x, y) = ||x||^2 + ||y||^2 - 2*x*y
    distmat = (
        torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    )
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)

    # Handle numerical errors (ensure non-negative)
    distmat = distmat.clamp(min=1e-12).sqrt()

    # Move to CPU for complex indexing operations if GPU memory is a concern,
    # but with A100 40GB, we can likely stay on GPU for moderate sizes.
    # However, the iterative expansion logic is often easier/safer in Numpy/CPU
    # for variable length lists, or we use masked tensors.
    # The standard implementation usually converts to Numpy for the set operations.
    original_dist = distmat.cpu().numpy()
    del distmat
    gc.collect()

    # Use the original distance for the final fusion
    # We only need the Query x Gallery block later, but we need the full matrix for calculations
    all_dist = original_dist

    # -------------------------------------------------------------------------
    # 2. k-Reciprocal Nearest Neighbors
    # -------------------------------------------------------------------------
    # Get top k1+1 neighbors (including self)
    # Sort indices by distance
    initial_rank = np.argsort(all_dist, axis=1)

    # Convert to standard python types for list processing
    initial_rank = initial_rank.astype(np.int32)

    nn_k1 = []
    for i in range(all_num):
        # Reciprocal neighbors
        # Forward: k1 neighbors of i
        forward_k1 = initial_rank[i, : k1 + 1]

        # Backward: Check if i is in the k1 neighbors of the forward neighbors
        backward_k1 = initial_rank[forward_k1, : k1 + 1]

        # Find intersection (reciprocity)
        # We want indices in forward_k1 where i appears in their backward lists
        rows, cols = np.where(backward_k1 == i)
        reciprocal_indices = forward_k1[rows]

        # k-Reciprocal Expansion (k2)
        # If a neighbor is reciprocal, add its k2 neighbors to the set
        pred_indices = reciprocal_indices
        if k2 > 1:
            # Get k2 neighbors for each reciprocal neighbor
            expanded_candidates = initial_rank[pred_indices, : k2 + 1]
            # Flatten
            expanded_candidates = np.unique(expanded_candidates.flatten())
            # Add to set
            pred_indices = np.union1d(pred_indices, expanded_candidates)

        nn_k1.append(pred_indices)

    # -------------------------------------------------------------------------
    # 3. Jaccard Distance Calculation
    # -------------------------------------------------------------------------
    # We compute Jaccard distance between the sets nn_k1[i] and nn_k1[j]
    # To do this efficiently, we use a vector expansion method (VQE).
    # We create a sparse-like representation where V[i, j] = 1 if j in nn_k1[i] (weighted).

    # Weighting scheme: Gaussian weight based on original distance
    V = np.zeros((all_num, all_num), dtype=np.float32)

    for i in range(all_num):
        neighbors = nn_k1[i]
        # Weight = exp(-dist)
        # We use the original distances
        dists = all_dist[i, neighbors]
        weights = np.exp(-dists)

        # Normalize weights to sum to 1 (conceptually similar to Local Query Expansion)
        # Note: Standard implementation usually doesn't strictly normalize to 1 for Jaccard,
        # but weights the binary presence.
        V[i, neighbors] = weights / np.sum(weights)

    # Jaccard Distance via Min/Max kernel
    # J(A, B) = 1 - sum(min(A, B)) / sum(max(A, B))
    # We can compute intersection (min) and union (max) efficiently via matrix ops if V was binary.
    # With weights:
    # Intersection: sum(min(V_i, V_j))
    # Union: sum(max(V_i, V_j))

    # Since V is sparse-ish, we can iterate query by query to save memory
    # Or use matrix multiplication for intersection if binary.
    # For weighted min/max, we have to be careful.

    # Optimization: The standard implementation simplifies Jaccard for re-ranking
    # to using the inverted index or simple intersection of sets.
    # Here we implement the robust version using the computed V matrix.

    # To avoid O(N^2) loop in python, we focus on the Query vs Gallery block.
    # However, Jaccard requires full vectors.

    # Let's use the efficient logic:
    # Jaccard(i, j) = 1 - (Intersection / Union)
    # Intersection[i, j] = sum_k min(V[i, k], V[j, k])
    # Union[i, j] = sum_k max(V[i, k], V[j, k])

    # Since computing full N*N Jaccard is expensive (10000^2 is 100M, doable but slow in python loop),
    # we proceed with a vectorized approximation or just the query-gallery block.

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    # We only compute for query rows to save time
    # V is (all_num, all_num)
    V_query = V[:query_num]  # (Q, all)
    V_all = V  # (All, all)

    # Intersection = V_query @ V_all.T (if binary).
    # For min/max, we construct the distance.
    # Efficient Jaccard on GPU using PyTorch

    V_query_t = torch.from_numpy(V_query).to(device)
    V_all_t = torch.from_numpy(V_all).to(device)

    # Intersection: min(a, b). Sum over k.
    # This is hard to vectorize as matrix mult.
    # Standard approximation: Intersection approx A dot B (if normalized).
    # Let's use the standard "k-reciprocal" trick:
    # The paper simplifies the Jaccard distance calculation using the min-min metric.
    # But often, simply using Cosine distance on the V vectors is a very strong proxy for Jaccard.
    # Let's stick to the definition:
    # dist_jaccard[i,j] = 1 - sum_k(min(Vi_k, Vj_k)) / sum_k(max(Vi_k, Vj_k))

    # GPU implementation of Generalized Jaccard
    # Expand dims: (Q, 1, N) and (1, All, N) -> Memory heavy (Q*All*N).
    # 2600 * 10000 * 10000 floats is too big.

    # Fallback to CPU loop with Numba or just efficient batching?
    # Or use the simplified inverted index form.

    # Given the constraints and the goal (MAP@5), we will use the robust intersection logic
    # but batched.

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        # Indices where V[i] is non-zero
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]

        # This loop is effectively iterating over neighbors of neighbors
        # Much faster than N*N
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]

        if len(indImages) > 0:
            indImages = np.hstack(indImages)
            indImages = np.unique(indImages)

            # Compute Jaccard for these candidates only
            # V[i] shape (N,), V[indImages] shape (M, N)

            # Intersection: sum(min(v_i, v_j))
            # We can just sum min over the non-zero columns of V[i]
            # Optimization: only check columns where V[i] > 0 (indNonZero)

            v_i_subset = V[i, indNonZero]  # (K,)
            v_candidates_subset = V[indImages][:, indNonZero]  # (M, K)

            # min(a, b) = 0.5 * (a + b - |a - b|)
            # intersection = sum(min)

            # Broadcasting
            # v_i_subset: (1, K)
            # v_candidates_subset: (M, K)

            abs_diff = np.abs(v_candidates_subset - v_i_subset)
            sum_min = 0.5 * np.sum(v_candidates_subset + v_i_subset - abs_diff, axis=1)

            # Union = sum(max) = sum(a) + sum(b) - intersection
            # sum(v_i) is constant for i
            # sum(v_candidates) can be precomputed

            sum_v_i = np.sum(v_i_subset)  # Scalar
            sum_v_candidates = np.sum(V[indImages], axis=1)  # (M,)

            sum_max = sum_v_i + sum_v_candidates - sum_min

            jaccard_dist[i, indImages] = 1 - (sum_min / (sum_max + 1e-8))

            # For non-candidates, distance is 1 (intersection 0)
            # We initialized with 0, so we need to set unvisited to 1?
            # Actually, standard logic: if no intersection, dist is 1.
            # But we initialized to 0. Let's fix.

    # However, initializing to 1 and updating is safer.
    final_jaccard = np.ones((query_num, all_num), dtype=np.float32)

    # We only computed for candidates, copy over
    # (The logic above computed 1 - Jaccard for candidates.
    # If we didn't touch it, it's 0. But it should be 1.)

    # Let's refine the loop logic to be strictly correct with initialization
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        if len(indImages) > 0:
            indImages = np.hstack(indImages)
            indImages = np.unique(indImages)

            v_i_subset = V[i, indNonZero]
            v_candidates_subset = V[indImages][:, indNonZero]

            abs_diff = np.abs(v_candidates_subset - v_i_subset)
            sum_min = 0.5 * np.sum(v_candidates_subset + v_i_subset - abs_diff, axis=1)

            sum_v_i = np.sum(V[i])
            sum_v_candidates = np.sum(V[indImages], axis=1)

            sum_max = sum_v_i + sum_v_candidates - sum_min

            final_jaccard[i, indImages] = 1 - (sum_min / (sum_max + 1e-8))

    # -------------------------------------------------------------------------
    # 4. Final Distance Fusion
    # -------------------------------------------------------------------------
    # Final = (1-lambda) * Jaccard + lambda * Original
    # We only need the Query -> Gallery part

    original_dist_sub = original_dist[:query_num, query_num:]
    jaccard_dist_sub = final_jaccard[:, query_num:]

    final_dist = (
        1 - lambda_value
    ) * jaccard_dist_sub + lambda_value * original_dist_sub

    return final_dist
