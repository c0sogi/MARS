import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


def l2_normalize(features):
    """
    L2 normalizes the features along the last dimension.

    Args:
        features (torch.Tensor or np.ndarray): Input features.

    Returns:
        torch.Tensor: Normalized features.
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    return F.normalize(features, p=2, dim=1)


def get_embeddings(model, loader, device, cache_stem, load_cached_data=True):
    """
    Extracts embeddings from a dataloader using the model.
    Implements caching to .npy files in Config.WORKING_DIR.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Data loader.
        device (torch.device): Device to run inference on.
        cache_stem (str): Unique identifier for the cache file (e.g., 'train_gallery').
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (embeddings (np.array), targets (np.array))
               targets will be labels (int) for train/val, or image_ids (str) for test.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    emb_path = os.path.join(Config.WORKING_DIR, f"{cache_stem}_emb.npy")
    tgt_path = os.path.join(Config.WORKING_DIR, f"{cache_stem}_tgt.npy")

    # 1. Try Load from Cache
    if load_cached_data and os.path.exists(emb_path) and os.path.exists(tgt_path):
        print(f"Loading cached embeddings from {emb_path}...")
        embeddings = np.load(emb_path)
        targets = np.load(tgt_path, allow_pickle=True)
        return embeddings, targets

    # 2. Compute from Scratch
    print(f"Computing embeddings for {cache_stem}...")
    model.eval()
    model.to(device)

    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {cache_stem}", leave=False):
            images, targets = batch
            images = images.to(device)

            # Forward pass (returns embeddings in eval mode)
            emb = model(images)

            all_embeddings.append(emb.cpu().numpy())

            # Handle targets (Tensor for labels, tuple/list for IDs)
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.cpu().numpy())
            else:
                all_targets.extend(targets)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    if isinstance(all_targets[0], np.ndarray) or isinstance(
        all_targets[0], (int, float)
    ):
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_targets = np.array(all_targets)

    # 3. Save to Cache
    np.save(emb_path, all_embeddings)
    np.save(tgt_path, all_targets)
    print(f"Saved embeddings to {emb_path}")

    return all_embeddings, all_targets


def compute_cosine_distance(query_feats, gallery_feats):
    """
    Computes cosine distance (1 - cosine_similarity) between query and gallery.
    Inputs are assumed to be un-normalized (normalization happens internally).

    Args:
        query_feats (np.array): (N_query, D)
        gallery_feats (np.array): (N_gallery, D)

    Returns:
        torch.Tensor: Distance matrix of shape (N_query, N_gallery)
    """
    q = l2_normalize(query_feats)
    g = l2_normalize(gallery_feats)

    # Cosine Similarity: Q * G^T
    # Range [-1, 1]
    sim = torch.mm(q, g.t())

    # Cosine Distance: 1 - Sim
    # Range [0, 2]
    dist = 1.0 - sim
    return dist


def perform_query_expansion(query_feats, gallery_feats, top_k=5, alpha=0.5):
    """
    Performs Query Expansion (QE).
    Replaces the query feature with a weighted average of itself and its top-k neighbors in the gallery.

    Args:
        query_feats (np.array): (N_query, D)
        gallery_feats (np.array): (N_gallery, D)
        top_k (int): Number of neighbors to use.
        alpha (float): Weight for the original query. 1.0 means no expansion.
                       Formula: new_q = alpha * q + (1-alpha) * mean(neighbors)

    Returns:
        np.array: Expanded query features (N_query, D)
    """
    # print(f"Performing Query Expansion (k={top_k})...")

    # Convert to torch for fast matrix ops
    q = l2_normalize(query_feats)
    g = l2_normalize(gallery_feats)

    # Compute Similarity
    sim = torch.mm(q, g.t())  # (N_q, N_g)

    # Get Top K indices
    _, indices = torch.topk(sim, k=top_k, dim=1)  # (N_q, k)

    # Gather neighbor features
    # indices is (N_q, k), we want to select rows from g (N_g, D)
    # result shape: (N_q, k, D)
    neighbors = g[indices]

    # Compute Mean of neighbors
    neighbors_mean = torch.mean(neighbors, dim=1)  # (N_q, D)

    # Combine
    expanded_q = alpha * q + (1.0 - alpha) * neighbors_mean

    # Renormalize
    expanded_q = l2_normalize(expanded_q)

    return expanded_q.cpu().numpy()


def perform_jaccard_reranking(query_feats, gallery_feats, k1=20, lambda_value=0.3):
    """
    Performs k-Reciprocal / Jaccard Re-ranking.
    Calculates Jaccard distance between the k-nearest neighbor sets of query and gallery.
    Combines with original cosine distance.

    Args:
        query_feats (np.array): Query embeddings.
        gallery_feats (np.array): Gallery embeddings.
        k1 (int): Number of neighbors to define the set.
        lambda_value (float): Weight for Jaccard distance (0 to 1).
                              Final = (1-lambda)*CosineDist + lambda*JaccardDist.

    Returns:
        np.array: Re-ranked distance matrix (N_query, N_gallery).
    """
    # print(f"Performing Jaccard Re-ranking (k={k1})...")

    # 1. Compute Base Distance (Cosine)
    # Use torch for GPU/CPU acceleration
    dist_qg = compute_cosine_distance(query_feats, gallery_feats)  # (Q, G)
    dist_gg = compute_cosine_distance(gallery_feats, gallery_feats)  # (G, G)

    num_q = dist_qg.shape[0]
    num_g = dist_qg.shape[1]

    # 2. Get Top-K Neighbor Indices
    # We want smallest distance
    _, indices_q = torch.topk(dist_qg, k=k1, dim=1, largest=False)  # (Q, k)
    _, indices_g = torch.topk(dist_gg, k=k1, dim=1, largest=False)  # (G, k)

    # 3. Construct Binary Adjacency Matrices (Sparse representation via dense tensors)
    # We represent the neighbor set as a binary vector of length |G|
    # This might be memory intensive if G is huge, but for 7k images it's fine.
    # 7000 * 7000 floats = 196MB.

    def get_binary_neighbor_matrix(indices, num_ref):
        # indices: (N, k)
        # output: (N, num_ref) binary
        B = torch.zeros(
            indices.shape[0], num_ref, device=indices.device, dtype=torch.float32
        )
        B.scatter_(1, indices, 1.0)
        return B

    # Move to same device as distances
    device = dist_qg.device
    indices_q = indices_q.to(device)
    indices_g = indices_g.to(device)

    V_q = get_binary_neighbor_matrix(indices_q, num_g)  # (Q, G)
    V_g = get_binary_neighbor_matrix(indices_g, num_g)  # (G, G)

    # 4. Compute Jaccard Distance
    # Intersection: V_q * V_g^T
    intersection = torch.mm(V_q, V_g.t())  # (Q, G)

    # Union: |V_q| + |V_g| - Intersection
    # Since V are binary, sum(dim=1) is just k1 (or close to it)
    sq_q = V_q.sum(dim=1, keepdim=True)  # (Q, 1)
    sq_g = V_g.sum(dim=1, keepdim=True).t()  # (1, G)

    union = sq_q + sq_g - intersection

    # Jaccard Similarity = Intersection / Union
    jaccard_sim = intersection / (union + 1e-6)

    # Jaccard Distance = 1 - Similarity
    jaccard_dist = 1.0 - jaccard_sim

    # 5. Combine Distances
    final_dist = (1.0 - lambda_value) * dist_qg + lambda_value * jaccard_dist

    return final_dist.cpu().numpy()


def generate_predictions(
    dist_matrix,
    test_ids,
    gallery_labels,
    label_encoder,
    threshold=0.5,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Generates the submission file based on the distance matrix and open-set threshold.

    Args:
        dist_matrix (np.array): (N_test, N_gallery) distance matrix.
        test_ids (np.array): Array of test image filenames.
        gallery_labels (np.array): Array of gallery label indices.
        label_encoder (LabelEncoder): Fitted encoder to map indices back to strings.
        threshold (float): Distance threshold for 'new_whale'.
        output_path (str): Path to save submission CSV.
    """
    print(f"Generating predictions with threshold={threshold}...")

    predictions = []

    # Iterate over each test query
    for i in range(len(test_ids)):
        query_dists = dist_matrix[i]  # Shape (N_gallery,)

        # Get all candidates sorted by distance
        # We only care about the top few to merge with new_whale
        # But to be safe, let's pick top 10 unique IDs

        sorted_indices = np.argsort(query_dists)

        candidates = []
        seen_ids = set()

        # Collect top known candidates
        for idx in sorted_indices:
            dist = query_dists[idx]
            label_idx = gallery_labels[idx]

            # Map to string ID
            try:
                str_id = label_encoder.inverse_transform([label_idx])[0]
            except:
                continue

            if str_id not in seen_ids:
                candidates.append((str_id, dist))
                seen_ids.add(str_id)

            if len(candidates) >= 5:
                break

        # Insert 'new_whale' as a virtual candidate with distance = threshold
        candidates.append(("new_whale", threshold))

        # Sort candidates by distance (ascending)
        # Stable sort to prefer known whales if distance is exactly equal to threshold
        candidates.sort(key=lambda x: x[1])

        # Select top 5 IDs
        top_5 = [c[0] for c in candidates[:5]]

        # Format string
        pred_str = " ".join(top_5)
        predictions.append(pred_str)

    # Create DataFrame
    df_sub = pd.DataFrame({"Image": test_ids, "Id": predictions})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference_pipeline(
    model,
    train_loader,
    test_loader,
    label_encoder,
    device=Config.DEVICE,
    load_cached=True,
):
    """
    Orchestrates the full inference pipeline:
    1. Extract Embeddings (Train & Test).
    2. Query Expansion on Test.
    3. Jaccard Re-ranking.
    4. Generate Submission.

    Args:
        model: Trained model.
        train_loader: Gallery loader.
        test_loader: Query loader.
        label_encoder: Fitted LabelEncoder.
        device: Torch device.
        load_cached: Whether to use cached embeddings.
    """
    seed_everything(Config.SEED)

    # 1. Extract Embeddings
    # Gallery (Train)
    gallery_emb, gallery_labels = get_embeddings(
        model, train_loader, device, "inf_gallery_train", load_cached_data=load_cached
    )

    # Query (Test)
    query_emb, query_ids = get_embeddings(
        model, test_loader, device, "inf_query_test", load_cached_data=load_cached
    )

    # 2. Query Expansion
    # Expand query features using the gallery
    query_emb_qe = perform_query_expansion(query_emb, gallery_emb, top_k=5, alpha=0.5)

    # 3. Re-ranking
    # Compute robust distances
    final_dists = perform_jaccard_reranking(
        query_emb_qe, gallery_emb, k1=20, lambda_value=0.3
    )

    # 4. Generate Submission
    # Threshold determines when 'new_whale' appears.
    # Cosine distance range [0, 2].
    # A threshold of 0.6 corresponds to Cosine Sim of 0.4.
    # This is a hyperparameter to tune.
    generate_predictions(
        final_dists,
        query_ids,
        gallery_labels,
        label_encoder,
        threshold=0.6,
        output_path=Config.SUBMISSION_PATH,
    )
