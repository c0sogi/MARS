import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


def load_class_mapping():
    """
    Loads the mapping from internal class index to original hotel_id.
    """
    cache_path = os.path.join(Config.working_dir, "classes.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    else:
        # Fallback: read from metadata if cache doesn't exist
        df = pd.read_csv(Config.train_metadata_path)
        return np.sort(df["hotel_id"].unique())


def database_augmentation(embeddings, top_k=5, chunk_size=1024):
    """
    Performs Database Augmentation (DBA) on the gallery embeddings.
    Replaces each embedding with a weighted average of itself and its nearest neighbors.

    Args:
        embeddings (torch.Tensor): Normalized gallery embeddings (N, D).
        top_k (int): Number of neighbors to aggregate.
        chunk_size (int): Batch size for processing to manage memory.

    Returns:
        torch.Tensor: Refined and normalized embeddings (N, D).
    """
    N, D = embeddings.shape
    refined_embeddings = torch.zeros_like(embeddings)

    # Ensure embeddings are on the correct device
    device = embeddings.device

    # Process in chunks
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        batch = embeddings[i:end]  # (B, D)

        # Compute similarity: (B, D) @ (D, N) -> (B, N)
        sim_matrix = torch.matmul(batch, embeddings.T)

        # Get top-k neighbors
        # values: (B, K), indices: (B, K)
        top_vals, top_inds = torch.topk(sim_matrix, k=top_k, dim=1)

        # Gather neighbor embeddings: (B, K, D)
        neighbors = embeddings[top_inds]

        # Calculate weights based on similarity
        # Clamp negative similarities to 0 for stability
        weights = torch.clamp(top_vals, min=0.0).unsqueeze(-1)  # (B, K, 1)

        # Weighted sum: (B, K, D) * (B, K, 1) -> sum over K -> (B, D)
        # We include the vector itself because it is always the top-1 neighbor (sim=1.0)
        weighted_sum = (neighbors * weights).sum(dim=1)

        refined_embeddings[i:end] = weighted_sum

    # L2 Normalize the result
    return F.normalize(refined_embeddings, dim=1)


def query_expansion(query_embeddings, gallery_embeddings, top_k=5, chunk_size=1024):
    """
    Performs Query Expansion (QE) on the query embeddings.
    Refines query embeddings by aggregating them with top-k retrieved gallery images.

    Args:
        query_embeddings (torch.Tensor): Normalized query embeddings (M, D).
        gallery_embeddings (torch.Tensor): Normalized gallery embeddings (N, D).
        top_k (int): Number of gallery neighbors to use.
        chunk_size (int): Batch size.

    Returns:
        torch.Tensor: Refined and normalized query embeddings (M, D).
    """
    M, D = query_embeddings.shape
    refined_queries = torch.zeros_like(query_embeddings)

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        q_batch = query_embeddings[i:end]  # (B, D)

        # Similarity: (B, D) @ (D, N) -> (B, N)
        sim_matrix = torch.matmul(q_batch, gallery_embeddings.T)

        # Get top-k gallery neighbors
        top_vals, top_inds = torch.topk(sim_matrix, k=top_k, dim=1)

        # Gather neighbors: (B, K, D)
        neighbors = gallery_embeddings[top_inds]

        # Weights
        weights = torch.clamp(top_vals, min=0.0).unsqueeze(-1)  # (B, K, 1)

        # Aggregate: Original query + weighted neighbors
        # Note: Depending on strategy, one might weight the original query differently.
        # Here we simply add the weighted sum of neighbors to the original query.
        neighbor_sum = (neighbors * weights).sum(dim=1)  # (B, D)

        refined_queries[i:end] = q_batch + neighbor_sum

    return F.normalize(refined_queries, dim=1)


def get_refined_embeddings(
    gallery_embeddings_np, query_embeddings_np, load_cached_data=True
):
    """
    Orchestrates the refinement process (DBA + QE) with caching.

    Args:
        gallery_embeddings_np (np.ndarray): Raw gallery embeddings.
        query_embeddings_np (np.ndarray): Raw query embeddings.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (refined_gallery_np, refined_query_np)
    """
    os.makedirs(Config.working_dir, exist_ok=True)

    gal_cache_path = os.path.join(Config.working_dir, "refined_gallery_embeddings.npy")
    qry_cache_path = os.path.join(Config.working_dir, "refined_query_embeddings.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(gal_cache_path)
        and os.path.exists(qry_cache_path)
    ):
        print("Loading refined embeddings from cache...")
        return np.load(gal_cache_path), np.load(qry_cache_path)

    # 2. Compute from scratch
    print("Computing refined embeddings (DBA + QE)...")

    device = Config.device
    # Convert to tensor and normalize
    gal_tensor = torch.from_numpy(gallery_embeddings_np).to(device)
    qry_tensor = torch.from_numpy(query_embeddings_np).to(device)

    gal_tensor = F.normalize(gal_tensor, dim=1)
    qry_tensor = F.normalize(qry_tensor, dim=1)

    # Apply Database Augmentation
    if Config.use_dba:
        print(f"Applying Database Augmentation (k={Config.dba_neighbors})...")
        gal_tensor = database_augmentation(gal_tensor, top_k=Config.dba_neighbors)

    # Apply Query Expansion
    if Config.use_qe:
        print(f"Applying Query Expansion (k={Config.dba_neighbors})...")
        qry_tensor = query_expansion(qry_tensor, gal_tensor, top_k=Config.dba_neighbors)

    refined_gal = gal_tensor.cpu().numpy()
    refined_qry = qry_tensor.cpu().numpy()

    # 3. Save to cache
    print("Saving refined embeddings to cache...")
    np.save(gal_cache_path, refined_gal)
    np.save(qry_cache_path, refined_qry)

    return refined_gal, refined_qry


def generate_predictions(
    query_embeddings, gallery_embeddings, gallery_labels, top_k=5, chunk_size=1024
):
    """
    Computes similarity and generates predictions.

    Args:
        query_embeddings (np.ndarray): (M, D)
        gallery_embeddings (np.ndarray): (N, D)
        gallery_labels (np.ndarray): (N,) Original hotel IDs for gallery images.
        top_k (int): Number of predictions per query.

    Returns:
        list: List of strings, where each string is space-delimited hotel IDs.
    """
    device = Config.device
    M = query_embeddings.shape[0]

    # Convert to tensor
    qry_tensor = torch.from_numpy(query_embeddings).to(device)
    gal_tensor = torch.from_numpy(gallery_embeddings).to(device)

    predictions = []

    print("Generating predictions...")
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        batch_q = qry_tensor[i:end]

        # Similarity
        sim = torch.matmul(batch_q, gal_tensor.T)

        # Top K indices
        _, indices = torch.topk(sim, k=top_k, dim=1)

        # Map indices to hotel IDs
        indices_np = indices.cpu().numpy()

        for row_idx in range(indices_np.shape[0]):
            # Get the gallery labels for the top k neighbors
            top_labels = gallery_labels[indices_np[row_idx]]

            # Format as space-delimited string
            pred_str = " ".join(map(str, top_labels))
            predictions.append(pred_str)

    return predictions


def run_post_processing(load_cached_data=True):
    """
    Main function to run the post-processing pipeline.
    Loads raw embeddings, refines them, generates predictions, and saves submission.
    """
    seed_everything(Config.seed)

    # 1. Load Data
    # Load raw embeddings (assumed to be generated by inference step)
    if not os.path.exists(Config.gallery_embeddings_path) or not os.path.exists(
        Config.query_embeddings_path
    ):
        print("Error: Raw embeddings not found. Run inference first.")
        return

    # Using pandas to read parquet then convert to numpy
    # Assuming parquet files have columns like 'emb_0', 'emb_1'... or just a single array column?
    # Usually simplest to save/load as numpy or parquet with specific schema.
    # Given the context, we'll assume standard reading.
    try:
        gallery_df = pd.read_parquet(Config.gallery_embeddings_path)
        query_df = pd.read_parquet(Config.query_embeddings_path)

        # Assuming the embedding is stored as a list/array in a column named 'embedding'
        # or all columns are dimensions.
        # Let's assume the parquet was saved such that values are the embedding dimensions.
        # If there's an 'image' column, drop it.
        if "image" in gallery_df.columns:
            gallery_embeddings = (
                np.stack(gallery_df["embedding"].values)
                if "embedding" in gallery_df.columns
                else gallery_df.drop(columns=["image"]).values
            )
        else:
            gallery_embeddings = gallery_df.values

        if "image" in query_df.columns:
            query_embeddings = (
                np.stack(query_df["embedding"].values)
                if "embedding" in query_df.columns
                else query_df.drop(columns=["image"]).values
            )
        else:
            query_embeddings = query_df.values

        # Ensure float32
        gallery_embeddings = gallery_embeddings.astype(np.float32)
        query_embeddings = query_embeddings.astype(np.float32)

    except Exception as e:
        print(f"Failed to load parquet embeddings: {e}")
        return

    # Load Metadata for labels and IDs
    train_df = pd.read_csv(Config.train_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # Gallery labels (original hotel_ids)
    # Important: The gallery embeddings must correspond row-wise to train_df
    gallery_labels = train_df["hotel_id"].values

    # Test image IDs
    test_image_ids = test_df["image"].values

    # 2. Refine Embeddings (DBA + QE)
    refined_gal, refined_qry = get_refined_embeddings(
        gallery_embeddings, query_embeddings, load_cached_data=load_cached_data
    )

    # 3. Generate Predictions
    preds = generate_predictions(refined_qry, refined_gal, gallery_labels, top_k=5)

    # 4. Save Submission
    submission_df = pd.DataFrame({"image": test_image_ids, "hotel_id": preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")

    # Validate shape
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())
