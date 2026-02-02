import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from library.config import Config
from library.dataset import HotelDataset, get_transforms
from library.model import HotelRecognitionModel
from library.engine import extract_embeddings
from library.utils import seed_everything


def load_or_extract_embeddings(
    model, dataset, device, cache_path, load_cached_data=True
):
    """
    Loads embeddings from cache if available; otherwise extracts them using the model.

    Args:
        model (nn.Module): The trained model.
        dataset (Dataset): The dataset to extract features from.
        device (str): Device to run inference on.
        cache_path (str): Path to save/load the .npy file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Embeddings matrix.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        try:
            embeddings = np.load(cache_path)
            return embeddings
        except Exception as e:
            print(f"Failed to load cache ({e}). Re-extracting...")

    # Extract features
    print(f"Extracting features for {len(dataset)} images...")
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    embeddings = extract_embeddings(dataloader, model, device)

    # Save to cache
    print(f"Saving embeddings to cache: {cache_path}")
    np.save(cache_path, embeddings)

    return embeddings


def perform_dba(embeddings, k=Config.KNN):
    """
    Performs Database Augmentation (DBA) on the gallery embeddings.
    Replaces each embedding with a weighted average of itself and its k nearest neighbors.

    Args:
        embeddings (np.ndarray): The gallery embeddings (N, D).
        k (int): Number of neighbors to use.

    Returns:
        np.ndarray: Refined embeddings.
    """
    if not Config.USE_DBA:
        return embeddings

    print(f"Performing Database Augmentation (DBA) with k={k}...")

    # L2 Normalize embeddings before search
    feats = normalize(embeddings, axis=1)

    # Find k nearest neighbors for each sample in the gallery
    # Using brute force or optimized algorithm depending on size
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine", n_jobs=-1)
    nbrs.fit(feats)
    distances, indices = nbrs.kneighbors(feats)

    # Aggregate embeddings
    # We use a simple average of the neighbors (including self)
    # Note: indices[:, 0] is the point itself

    refined_feats = np.zeros_like(embeddings)

    # Vectorized accumulation
    # For very large datasets, a loop might be necessary to save memory,
    # but 70k x 512 fits easily in RAM.
    for i in range(len(embeddings)):
        neighbor_indices = indices[i]
        # Fetch neighbor vectors
        neighbor_feats = embeddings[neighbor_indices]
        # Compute mean
        refined_feats[i] = np.mean(neighbor_feats, axis=0)

    # Renormalize
    refined_feats = normalize(refined_feats, axis=1)

    return refined_feats


def perform_qe(query_embeddings, gallery_embeddings, k=Config.KNN):
    """
    Performs Query Expansion (QE).
    Refines query embeddings by aggregating them with their top-k retrieved gallery items.

    Args:
        query_embeddings (np.ndarray): Query embeddings (M, D).
        gallery_embeddings (np.ndarray): Refined gallery embeddings (N, D).
        k (int): Number of neighbors.

    Returns:
        np.ndarray: Refined query embeddings.
    """
    if not Config.USE_QE:
        return query_embeddings

    print(f"Performing Query Expansion (QE) with k={k}...")

    # Normalize
    q_feats = normalize(query_embeddings, axis=1)
    g_feats = normalize(gallery_embeddings, axis=1)

    # Fit NN on Gallery
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine", n_jobs=-1)
    nbrs.fit(g_feats)

    # Find neighbors for queries
    distances, indices = nbrs.kneighbors(q_feats)

    refined_queries = np.zeros_like(query_embeddings)

    for i in range(len(query_embeddings)):
        # Original query
        q_vec = query_embeddings[i]
        # Retrieved neighbors
        neighbor_indices = indices[i]
        neighbor_vecs = gallery_embeddings[neighbor_indices]

        # Combine: Average of Query + Neighbors
        # Stack query and neighbors then mean
        combined = np.vstack([q_vec, neighbor_vecs])
        refined_queries[i] = np.mean(combined, axis=0)

    # Renormalize
    refined_queries = normalize(refined_queries, axis=1)

    return refined_queries


def generate_predictions(query_embeddings, gallery_embeddings, gallery_labels, top_k=5):
    """
    Calculates similarity and generates top-k predictions.

    Args:
        query_embeddings (np.ndarray): (M, D)
        gallery_embeddings (np.ndarray): (N, D)
        gallery_labels (np.ndarray): (N,) Array of hotel_ids corresponding to gallery rows.
        top_k (int): Number of predictions per query.

    Returns:
        list: List of strings, where each string is a space-delimited list of hotel_ids.
    """
    print("Generating predictions...")

    # Normalize
    Q = normalize(query_embeddings, axis=1)
    G = normalize(gallery_embeddings, axis=1)

    # Compute Cosine Similarity: S = Q @ G.T
    # Using matrix multiplication on GPU if possible, otherwise CPU
    # Given 220GB RAM, CPU is safe. GPU is faster.

    device = Config.DEVICE
    Q_t = torch.from_numpy(Q).to(device)
    G_t = torch.from_numpy(G).to(device)

    # Chunking to prevent OOM on GPU if G is very large, though 70k is manageable on A100
    # 70,000 * 10,000 * 4 bytes approx 2.8 GB. Safe.

    similarity = torch.matmul(Q_t, G_t.T)  # (M, N)

    # Get Top K indices
    values, indices = torch.topk(similarity, k=top_k, dim=1, largest=True, sorted=True)

    indices = indices.cpu().numpy()

    preds = []
    for idx_row in indices:
        # Map indices to hotel_ids
        top_hotels = gallery_labels[idx_row]
        # Convert to space-delimited string
        pred_str = " ".join(map(str, top_hotels))
        preds.append(pred_str)

    return preds


def run_inference(load_cached_data=True):
    """
    Main inference pipeline.

    Args:
        load_cached_data (bool): If True, reuses extracted embeddings from disk.
    """
    seed_everything(Config.SEED)

    # 1. Setup Data
    # Gallery (Train Set)
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Missing training metadata: {Config.TRAIN_METADATA_PATH}"
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    gallery_labels = train_df["hotel_id"].values

    # Query (Test Set)
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(f"Missing test metadata: {Config.TEST_METADATA_PATH}")

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    query_image_names = test_df["image"].values

    # Datasets
    # We use is_test=True for Gallery to avoid label processing overhead in Dataset class
    # and to ensure deterministic transforms (Validation transform).
    gallery_dataset = HotelDataset(
        Config.TRAIN_METADATA_PATH, transform=get_transforms(mode="test"), is_test=True
    )

    query_dataset = HotelDataset(
        Config.TEST_METADATA_PATH, transform=get_transforms(mode="test"), is_test=True
    )

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = HotelRecognitionModel()
    model.to(device)

    # Load weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model weights from {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Best model not found at {Config.BEST_MODEL_PATH}. Using random weights (or pre-trained backbone)."
        )

    model.eval()

    # 3. Extract Features
    gallery_embeddings = load_or_extract_embeddings(
        model, gallery_dataset, device, Config.GALLERY_EMBEDDINGS_PATH, load_cached_data
    )

    query_embeddings = load_or_extract_embeddings(
        model, query_dataset, device, Config.QUERY_EMBEDDINGS_PATH, load_cached_data
    )

    # 4. Graph-Based Regularization
    # DBA
    refined_gallery = perform_dba(gallery_embeddings, k=Config.KNN)

    # QE
    refined_query = perform_qe(query_embeddings, refined_gallery, k=Config.KNN)

    # 5. Generate Predictions
    predictions = generate_predictions(
        refined_query, refined_gallery, gallery_labels, top_k=Config.TOP_K
    )

    # 6. Create Submission
    submission_df = pd.DataFrame({"image": query_image_names, "hotel_id": predictions})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Validation check on output
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())
