import torch
import numpy as np
from library.config import seed_everything


def map_at_5(predictions, ground_truth):
    """
    Calculates the Mean Average Precision at 5 (MAP@5).

    For a single ground truth label per sample, this is equivalent to
    Mean Reciprocal Rank (MRR) at 5.

    Args:
        predictions (list of list): A list where each element is a list of
                                    predicted labels (strings), ordered by confidence.
        ground_truth (list): A list of the true labels (strings) for each sample.

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs ground_truth ({len(ground_truth)})"
        )

    score = 0.0
    num_samples = len(ground_truth)

    for i in range(num_samples):
        # Consider only the top 5 predictions
        preds = predictions[i][:5]
        truth = ground_truth[i]

        if truth in preds:
            # Rank is 0-indexed
            rank = preds.index(truth)
            # Precision at k is 1 / (rank + 1)
            score += 1.0 / (rank + 1)

    return score / num_samples


def k_reciprocal_re_ranking(prob_feat, gal_feat, k1=20, k2=6, lambda_value=0.3):
    """
    Computes re-ranked distances using the Jaccard distance of k-nearest neighbor sets.
    This exploits the manifold structure of the data to refine similarity.

    Args:
        prob_feat (numpy.ndarray or torch.Tensor): Query/Probe embeddings of shape (N, D).
        gal_feat (numpy.ndarray or torch.Tensor): Gallery embeddings of shape (M, D).
        k1 (int): Number of neighbors to consider for the Jaccard set.
        k2 (int): Unused in this vectorized implementation (kept for API compatibility).
        lambda_value (float): Weighting factor for the Jaccard distance (0 to 1).
                              Final Dist = (1 - lambda) * Original + lambda * Jaccard.

    Returns:
        numpy.ndarray: The re-ranked distance matrix of shape (N, M).
    """
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch tensors on the correct device
    if isinstance(prob_feat, np.ndarray):
        prob_feat = torch.from_numpy(prob_feat)
    if isinstance(gal_feat, np.ndarray):
        gal_feat = torch.from_numpy(gal_feat)

    prob_feat = prob_feat.to(device).float()
    gal_feat = gal_feat.to(device).float()

    # L2 Normalize features (essential for Cosine/Euclidean equivalence)
    prob_feat = torch.nn.functional.normalize(prob_feat, p=2, dim=1)
    gal_feat = torch.nn.functional.normalize(gal_feat, p=2, dim=1)

    # -------------------------------------------------------------------------
    # 1. Compute Original Distances (Euclidean)
    # -------------------------------------------------------------------------
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y> = 2 - 2<x, y> (since normalized)

    # Query-Gallery Distance
    sim_qg = torch.matmul(prob_feat, gal_feat.t())
    dist_qg = 2.0 - 2.0 * sim_qg
    dist_qg = torch.clamp(dist_qg, min=0.0)  # Numerical stability

    # Gallery-Gallery Distance (needed for neighbors of gallery items)
    sim_gg = torch.matmul(gal_feat, gal_feat.t())
    dist_gg = 2.0 - 2.0 * sim_gg
    dist_gg = torch.clamp(dist_gg, min=0.0)

    # -------------------------------------------------------------------------
    # 2. Find k-Nearest Neighbors
    # -------------------------------------------------------------------------
    # Get indices of top-k neighbors (smallest distance)
    # We use k1 for the set size

    # Neighbors of Query in Gallery
    _, indices_qg = torch.topk(dist_qg, k=k1, dim=1, largest=False)  # (N, k1)

    # Neighbors of Gallery in Gallery
    _, indices_gg = torch.topk(dist_gg, k=k1, dim=1, largest=False)  # (M, k1)

    # -------------------------------------------------------------------------
    # 3. Compute Jaccard Distance
    # -------------------------------------------------------------------------
    # Jaccard(A, B) = 1 - |A intersect B| / |A union B|
    # A = Neighbors(Query), B = Neighbors(Gallery)

    N = prob_feat.size(0)
    M = gal_feat.size(0)

    # Construct sparse binary masks representing neighbor sets
    # mask[i, j] = 1.0 if j is in neighbors(i)

    # Mask for Query neighbors
    mask_q = torch.zeros((N, M), device=device, dtype=torch.float32)
    mask_q.scatter_(1, indices_qg, 1.0)

    # Mask for Gallery neighbors
    mask_g = torch.zeros((M, M), device=device, dtype=torch.float32)
    mask_g.scatter_(1, indices_gg, 1.0)

    # Intersection: Dot product of binary vectors
    # (N, M) x (M, M).T -> (N, M)
    intersection = torch.matmul(mask_q, mask_g.t())

    # Union: |A| + |B| - Intersection
    # |A| = k1, |B| = k1 (fixed size sets)
    union = k1 + k1 - intersection

    # Jaccard Distance
    jaccard_dist = 1.0 - (intersection / (union + 1e-8))

    # -------------------------------------------------------------------------
    # 4. Combine Distances
    # -------------------------------------------------------------------------
    # Combine original distance with Jaccard distance
    # Note: dist_qg is roughly in [0, 4], jaccard_dist is in [0, 1]
    # The lambda_value balances these.

    final_dist = (1.0 - lambda_value) * dist_qg + lambda_value * jaccard_dist

    return final_dist.cpu().numpy()
