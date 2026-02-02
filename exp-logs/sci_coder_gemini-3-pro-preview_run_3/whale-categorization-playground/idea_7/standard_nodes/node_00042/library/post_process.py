import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config


def compute_distance_matrix(input1, input2, metric="euclidean"):
    """
    Computes the distance matrix between two sets of vectors.

    Args:
        input1 (torch.Tensor): Shape (M, D)
        input2 (torch.Tensor): Shape (N, D)
        metric (str): 'euclidean' or 'cosine'

    Returns:
        torch.Tensor: Distance matrix of shape (M, N)
    """
    # Ensure inputs are on the same device
    if input1.device != input2.device:
        input2 = input2.to(input1.device)

    m, n = input1.size(0), input2.size(0)

    if metric == "euclidean":
        # (x-y)^2 = x^2 + y^2 - 2xy
        dist = (
            torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(input1, input2.t(), beta=1, alpha=-2)
        # Clamp to avoid negative values due to numerical precision
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    elif metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        input1_norm = F.normalize(input1, p=2, dim=1)
        input2_norm = F.normalize(input2, p=2, dim=1)
        cosine_sim = torch.mm(input1_norm, input2_norm.t())
        return 1.0 - cosine_sim

    else:
        raise ValueError(f"Unknown metric: {metric}")


def query_expansion(query_feats, gallery_feats, top_k=5, alpha=0.5):
    """
    Applies Query Expansion (QE) to refine query embeddings.

    Strategy:
    1. Retrieve top-k nearest neighbors from the gallery for each query.
    2. Compute a weighted average of the query and its neighbors.
    3. Re-normalize the expanded query.

    Args:
        query_feats (torch.Tensor): Query embeddings (N_q, D).
        gallery_feats (torch.Tensor): Gallery embeddings (N_g, D).
        top_k (int): Number of neighbors to use for expansion.
        alpha (float): Weight for the original query. (Currently implicit in mean).

    Returns:
        torch.Tensor: Expanded query embeddings (N_q, D).
    """
    device = query_feats.device
    gallery_feats = gallery_feats.to(device)

    # Normalize features for cosine similarity
    q_norm = F.normalize(query_feats, p=2, dim=1)
    g_norm = F.normalize(gallery_feats, p=2, dim=1)

    # Compute Cosine Similarity
    # Shape: (N_q, N_g)
    sim_mat = torch.mm(q_norm, g_norm.t())

    # Get top-k neighbors
    # indices shape: (N_q, top_k)
    _, indices = torch.topk(sim_mat, k=top_k, dim=1)

    # Gather neighbor features
    # indices needs to be expanded to gather from gallery_feats
    # gallery_feats: (N_g, D)
    # We want (N_q, top_k, D)

    # Efficient gathering via indexing
    # Create a tensor of expanded queries
    expanded_queries = torch.zeros_like(query_feats)

    for i in range(query_feats.size(0)):
        # Get neighbors for query i
        neighbor_indices = indices[i]
        neighbors = gallery_feats[neighbor_indices]  # (top_k, D)

        # Average strategy: (Query + Sum(Neighbors)) / (K + 1)
        # Or weighted average. Here we use simple sum fusion followed by norm.
        # This is equivalent to mean pooling in direction.
        fused_feat = query_feats[i] + neighbors.sum(dim=0)
        expanded_queries[i] = fused_feat

    # Re-normalize
    expanded_queries = F.normalize(expanded_queries, p=2, dim=1)

    return expanded_queries


def k_reciprocal_rerank(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    Applies k-Reciprocal Re-ranking to the distance matrix.

    Reference: Zhong et al. "Re-ranking Person Re-identification with k-reciprocal Encoding".

    This method computes a new distance metric based on the Jaccard distance
    between the k-reciprocal neighbor sets of the query and gallery.

    Args:
        probFea (torch.Tensor): Query features (N_q, D).
        galFea (torch.Tensor): Gallery features (N_g, D).
        k1 (int): The size of the k-reciprocal set.
        k2 (int): The size of the local query expansion for the k-reciprocal set.
        lambda_value (float): Weighting factor between original distance and Jaccard distance.

    Returns:
        torch.Tensor: Re-ranked distance matrix (N_q, N_g).
    """
    # Ensure evaluation mode and device consistency
    device = probFea.device
    galFea = galFea.to(device)

    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)

    # Concatenate all features to compute the global distance matrix
    # This is necessary because reciprocal neighbors can be within the gallery or query set
    feat = torch.cat([probFea, galFea], dim=0)

    # 1. Compute original Euclidean distance
    # Shape: (all_num, all_num)
    original_dist = compute_distance_matrix(feat, feat, metric="euclidean")

    # Normalize original distance to [0, 1] for combination later
    original_dist = original_dist / (original_dist.max() + 1e-8)

    # Transpose to CPU for complex indexing operations if GPU memory is tight,
    # but for Whale dataset sizes (approx 10k total), GPU is fine and faster.
    # We will stick to GPU tensors.

    # 2. Get initial ranking (top k1+1 to include self)
    # Shape: (all_num, all_num)
    # We only need the indices for the logic
    _, initial_rank = torch.topk(original_dist, k=all_num, dim=1, largest=False)

    # 3. Compute k-reciprocal sets and Jaccard distance
    # We will construct a weighted sparse matrix V where V[i, j] indicates
    # the trust/weight of neighbor j for sample i.

    # This part is complex to vectorize fully without massive memory usage for V (all_num x all_num).
    # Given N ~ 10,000, N^2 is 100M floats = 400MB. This fits easily on A100 GPU.

    # Initialize V matrix (all_num, all_num)
    V = torch.zeros_like(original_dist).to(device)

    # Iterate over each sample to determine its k-reciprocal set
    # Note: A fully vectorized implementation of k-reciprocal check is possible but obscure.
    # A loop over N is acceptable here given N=10k and efficient inner ops.

    # Convert to numpy for set operations if needed, but torch is preferred for speed.
    # We'll implement a semi-vectorized loop.

    initial_rank_np = initial_rank.cpu().numpy()
    original_dist_np = original_dist.cpu().numpy()

    # We will build V row by row
    for i in range(all_num):
        # Forward k1 neighbors
        forward_k1 = initial_rank_np[i, : k1 + 1]

        # Backward k1 neighbors for each forward neighbor
        backward_k1 = initial_rank_np[forward_k1, : k1 + 1]

        # Find reciprocal neighbors: those who have 'i' in their top k1
        # np.where returns (row_indices, col_indices)
        fi = np.where(backward_k1 == i)[0]

        # Get the actual indices of reciprocal neighbors
        k_reciprocal_idx = forward_k1[fi]

        # Calculate k-reciprocal expansion (k2)
        k_reciprocal_expansion_idx = k_reciprocal_idx
        for candidate in k_reciprocal_idx:
            # Candidates from the k-reciprocal set
            candidate_forward_k1 = initial_rank_np[candidate, : k1 // 2 + 1]

            # Check overlap condition
            # If overlap between candidate's neighbors and i's reciprocal set is significant
            candidate_backward_k1 = initial_rank_np[candidate_forward_k1, : k1 // 2 + 1]
            fi_candidate = np.where(candidate_backward_k1 == candidate)[0]
            candidate_reciprocal_idx = candidate_forward_k1[fi_candidate]

            if len(
                np.intersect1d(candidate_reciprocal_idx, k_reciprocal_idx)
            ) > 2 / 3 * len(candidate_reciprocal_idx):
                k_reciprocal_expansion_idx = np.append(
                    k_reciprocal_expansion_idx, candidate_forward_k1
                )

        k_reciprocal_expansion_idx = np.unique(k_reciprocal_expansion_idx)

        # Assign weights
        # Weight = exp(-dist)
        weights = np.exp(-original_dist_np[i, k_reciprocal_expansion_idx])

        # Update V matrix
        V[i, k_reciprocal_expansion_idx] = torch.from_numpy(weights).to(device)

    # 4. Jaccard Distance Calculation
    # Jaccard(A, B) = 1 - (Intersection / Union)
    # Generalized Jaccard for weighted vectors: 1 - sum(min(a,b)) / sum(max(a,b))

    # We focus on Query vs Gallery part
    # Split V into Query part and Gallery part
    V_query = V[:query_num]  # (N_q, all_num)
    V_gallery = V[query_num:]  # (N_g, all_num)

    # To compute Jaccard efficiently for all pairs (q, g):
    # Intersection: sum(min(V_q, V_g))
    # Union: sum(max(V_q, V_g))
    # This is computationally expensive (N_q * N_g * N_all).
    # Simplification used in standard ReID: Use k-reciprocal encoding distance.
    # However, with GPU, we can try a slightly optimized approach or standard simplified Jaccard.

    # Standard simplified Jaccard in ReID code often uses simple intersection logic
    # or just matrix multiplication if weights are binary.
    # Here weights are continuous.

    # Optimization: Jaccard distance can be computed via:
    # I = min(V_q, V_g) -> This is hard to matrix multiply.
    # Alternative: Use the intersection of indices.

    # Let's use the standard implementation logic which essentially computes:
    # dist_jaccard[i, j] = 1 - sum(min(V[i], V[j])) / sum(max(V[i], V[j]))

    # Given the constraints and potential runtime, we will use a simplified version
    # where we only consider the overlap of the non-zero elements (indices).
    # But to be precise, we perform the calculation.

    jaccard_dist = torch.zeros(query_num, galFea.size(0), device=device)

    # We iterate over queries to save memory if needed, but batching is better.
    # Let's batch queries.
    batch_size = 20
    for i in range(0, query_num, batch_size):
        end = min(i + batch_size, query_num)
        # Shape: (Batch, all_num)
        v_q_batch = V_query[i:end].unsqueeze(1)  # (Batch, 1, all_num)
        v_g_all = V_gallery.unsqueeze(0)  # (1, N_g, all_num)

        # Broadcasting is too heavy: (100, 7000, 10000) floats -> 7GB memory.
        # A100 has 40GB, so this is actually feasible!

        # Intersection: min(a, b)
        intersection = torch.min(v_q_batch, v_g_all).sum(dim=2)

        # Union: max(a, b)
        union = torch.max(v_q_batch, v_g_all).sum(dim=2)

        jaccard_batch = 1.0 - (intersection / (union + 1e-8))
        jaccard_dist[i:end] = jaccard_batch

    # 5. Final Distance Combination
    # Only take the relevant sub-matrix from original distance
    original_dist_qg = original_dist[:query_num, query_num:]

    final_dist = lambda_value * original_dist_qg + (1 - lambda_value) * jaccard_dist

    return final_dist
