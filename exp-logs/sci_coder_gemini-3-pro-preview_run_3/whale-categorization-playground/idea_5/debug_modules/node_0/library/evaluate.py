import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.rerank import re_ranking


def extract_embeddings(model, loader, device):
    """
    Extracts embeddings and targets (labels or filenames) from a loader.

    Args:
        model: The PyTorch model.
        loader: The DataLoader.
        device: The torch device.

    Returns:
        embeddings (np.ndarray): Normalized embeddings (N, D).
        targets (np.ndarray): Targets (N,) - either int labels or string filenames.
    """
    model.eval()
    embeddings_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            # Forward pass with label=None returns normalized embeddings
            emb = model(images, label=None)
            embeddings_list.append(emb.cpu().numpy())

            # Handle targets: Tensor (train/val labels) or Tuple (test filenames)
            if isinstance(targets, torch.Tensor):
                targets_list.append(targets.cpu().numpy())
            else:
                # targets is a tuple of strings
                targets_list.extend(targets)

    # Concatenate embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)

    # Process targets
    if len(targets_list) > 0:
        if isinstance(targets_list[0], np.ndarray) or isinstance(
            targets_list[0], (int, np.integer)
        ):
            targets = np.concatenate(targets_list, axis=0)
        else:
            targets = np.array(targets_list)
    else:
        targets = np.array([])

    return embeddings, targets


def get_cached_data(func, cache_key, load_cached_data, *args):
    """
    Wrapper to handle caching of embedding extraction.

    Args:
        func: The function to call if cache is missing (extract_embeddings).
        cache_key: Unique identifier for the cache file (e.g., 'train_gallery').
        load_cached_data: Boolean flag to enable/disable loading from cache.
        *args: Arguments to pass to func.

    Returns:
        embeddings, targets
    """
    cache_emb_path = os.path.join(Config.CACHE_DIR, f"{cache_key}_emb.npy")
    cache_tgt_path = os.path.join(Config.CACHE_DIR, f"{cache_key}_tgt.npy")

    if (
        load_cached_data
        and os.path.exists(cache_emb_path)
        and os.path.exists(cache_tgt_path)
    ):
        print(f"Loading cached features from {cache_key}...")
        embeddings = np.load(cache_emb_path)
        targets = np.load(cache_tgt_path, allow_pickle=True)
        return embeddings, targets

    print(f"Computing features for {cache_key}...")
    embeddings, targets = func(*args)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_emb_path, embeddings)
    np.save(cache_tgt_path, targets)

    return embeddings, targets


def validate(model, train_loader, val_loader, device, load_cached_data=False):
    """
    Performs retrieval-based validation (MAP@5).
    Treats Validation set as Queries and Training set as Gallery.
    """
    seed_everything(Config.SEED)

    # Extract features
    # Note: We use distinct cache keys to avoid confusion if loaders change
    train_emb, train_labels = get_cached_data(
        extract_embeddings,
        "val_gallery_train",
        load_cached_data,
        model,
        train_loader,
        device,
    )
    val_emb, val_labels = get_cached_data(
        extract_embeddings, "val_query_val", load_cached_data, model, val_loader, device
    )

    print(f"Validation: {len(val_emb)} queries, {len(train_emb)} gallery items.")

    # Compute Cosine Similarity Matrix (Queries x Gallery)
    # Embeddings are already L2 normalized by the model
    # Matrix multiplication: (Q, D) @ (G, D).T -> (Q, G)
    sim_matrix = np.dot(val_emb, train_emb.T)

    # Retrieve Top 5
    top_k = 5
    n_queries = len(val_labels)
    score_sum = 0.0

    # Iterate through queries to calculate MAP
    # We use argpartition for efficiency if G is large, but for ~6k sort is fine
    # We need descending order of similarity

    for i in range(n_queries):
        # Get similarities for query i
        sims = sim_matrix[i]

        # Get indices of top 5 (descending)
        # argsort returns ascending, so we take tail and reverse
        top_indices = np.argsort(sims)[-top_k:][::-1]

        true_label = val_labels[i]
        pred_labels = train_labels[top_indices]

        # Calculate AP@5
        if true_label in pred_labels:
            # Find rank (0-indexed)
            rank = np.where(pred_labels == true_label)[0][0]
            score_sum += 1.0 / (rank + 1)

    map5 = score_sum / n_queries
    print(f"Validation MAP@5: {map5:.10f}")

    return map5


def inference(model, train_loader, test_loader, device, load_cached_data=True):
    """
    Performs inference on the Test set using Training set as Gallery.
    Applies Re-ranking and generates submission.csv.
    """
    seed_everything(Config.SEED)

    # 1. Load Class Map (created by dataset.py)
    classes_path = os.path.join(Config.CACHE_DIR, "classes.npy")
    if not os.path.exists(classes_path):
        # Fallback: try to reconstruct or fail. dataset.py should have created it.
        print(
            "Warning: classes.npy not found in cache. Ensure get_loaders has been run."
        )
        return

    class_names = np.load(classes_path, allow_pickle=True)

    # 2. Extract Features
    train_emb, train_labels = get_cached_data(
        extract_embeddings,
        "inf_gallery_train",
        load_cached_data,
        model,
        train_loader,
        device,
    )
    test_emb, test_filenames = get_cached_data(
        extract_embeddings,
        "inf_query_test",
        load_cached_data,
        model,
        test_loader,
        device,
    )

    print(f"Inference: {len(test_emb)} queries, {len(train_emb)} gallery items.")

    # 3. Compute Distance Matrix with Re-ranking
    # Returns distance matrix (lower is better)
    print("Computing re-ranking distance matrix...")
    dist_matrix = re_ranking(
        test_emb,
        train_emb,
        k1=Config.RERANK_K1,
        k2=Config.RERANK_K2,
        lambda_value=Config.RERANK_LAMBDA,
    )

    # 4. Generate Predictions
    print("Generating predictions...")
    submission_rows = []

    for i in range(len(test_filenames)):
        fname = test_filenames[i]
        dists = dist_matrix[i]

        # Sort indices by distance (ascending)
        sorted_indices = np.argsort(dists)

        # Distance to the nearest neighbor
        min_dist = dists[sorted_indices[0]]

        # Collect top 5 unique known whale IDs
        top_known_ids = []
        seen_ids = set()

        for idx in sorted_indices:
            # Map gallery index -> label index -> string ID
            label_idx = train_labels[idx]
            whale_id = class_names[label_idx]

            if whale_id not in seen_ids:
                top_known_ids.append(whale_id)
                seen_ids.add(whale_id)

            if len(top_known_ids) >= 5:
                break

        # Construct Final Prediction List with 'new_whale'
        # Strategy:
        # If min_dist > Threshold, we are unsure -> Predict 'new_whale' first.
        # Else, we are confident -> Predict nearest known whale first, then 'new_whale'.

        final_preds = []
        if min_dist > Config.NEW_WHALE_THRESH:
            final_preds.append("new_whale")
            final_preds.extend(top_known_ids[:4])
        else:
            final_preds.append(top_known_ids[0])
            final_preds.append("new_whale")
            final_preds.extend(top_known_ids[1:4])

        # Join with spaces
        pred_str = " ".join(final_preds)
        submission_rows.append([fname, pred_str])

    # 5. Save Submission
    df_sub = pd.DataFrame(submission_rows, columns=["Image", "Id"])
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
