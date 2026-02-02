import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import HotelDataset, get_transforms, get_label_mapping
from library.utils import seed_everything


def extract_embeddings(
    model, dataloader, device, cache_path_emb, cache_path_ids, load_cached_data=True
):
    """
    Extracts embeddings from the model for a given dataloader.
    Handles caching of embeddings and identifiers (labels or image names).

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader containing images.
        device (torch.device): Device to run the model on.
        cache_path_emb (str): Path to save/load embeddings .npy file.
        cache_path_ids (str): Path to save/load identifiers .npy file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (embeddings, identifiers) as numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path_emb), exist_ok=True)

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_emb)
        and os.path.exists(cache_path_ids)
    ):
        print(f"Loading cached embeddings from {cache_path_emb}")
        embeddings = np.load(cache_path_emb)
        identifiers = np.load(cache_path_ids)
        return embeddings, identifiers

    # 2. Compute from scratch
    print(f"Computing embeddings (Cache miss or force reload)...")
    model.eval()
    model.to(device)

    emb_list = []
    id_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch: (images, labels) or (images, image_names)
            if len(batch) == 2:
                images, targets = batch
            else:
                images = batch[0]
                targets = None

            images = images.to(device)

            # Forward pass: get embeddings (labels=None)
            # Output shape: (Batch, EmbeddingDim)
            features = model(images, labels=None)

            # L2 Normalize features for Cosine Similarity
            features = F.normalize(features, p=2, dim=1)

            emb_list.append(features.cpu().numpy())

            # Collect identifiers
            if targets is not None:
                if isinstance(targets, torch.Tensor):
                    id_list.append(targets.cpu().numpy())
                else:
                    # targets is a tuple/list of strings (image names)
                    id_list.extend(targets)

    # Concatenate results
    embeddings = np.concatenate(emb_list, axis=0)

    if len(id_list) > 0:
        if isinstance(id_list[0], np.ndarray):
            identifiers = np.concatenate(id_list, axis=0)
        else:
            identifiers = np.array(id_list)
    else:
        identifiers = np.array([])

    # 3. Save to cache
    print(f"Saving embeddings to {cache_path_emb}")
    np.save(cache_path_emb, embeddings)
    np.save(cache_path_ids, identifiers)

    return embeddings, identifiers


def predict(
    model, test_loader, device=Config.DEVICE, load_cached_data=True, debug=Config.DEBUG
):
    """
    Main inference function. Generates gallery and query embeddings,
    performs retrieval, and creates the submission file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set (Query).
        device (torch.device): Computation device.
        load_cached_data (bool): Whether to use cached embeddings.
        debug (bool): If True, uses a subset of the gallery for faster processing.
    """
    seed_everything(Config.SEED)
    model.to(device)
    model.eval()

    # --- 1. Gallery Generation (Training Set) ---
    # We need to construct the gallery loader here
    print("Preparing Gallery...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Apply Debugging if requested
    if debug:
        print(f"Debug mode: Subsampling gallery to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Map hotel_id to label_idx
    id_to_idx, idx_to_id = get_label_mapping(load_cached_data=True)
    train_df["label_idx"] = train_df["hotel_id"].map(id_to_idx)

    # Drop any rows where mapping might have failed (safety check)
    train_df = train_df.dropna(subset=["label_idx"])
    train_df["label_idx"] = train_df["label_idx"].astype(int)

    # Create Gallery Dataset/Loader
    # Use 'val' transforms (no augmentation) for deterministic gallery embeddings
    gallery_dataset = HotelDataset(
        train_df, transform=get_transforms(mode="val"), mode="train"
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Extract Gallery Embeddings
    gallery_emb_path = Config.GALLERY_EMB_PATH
    gallery_ids_path = Config.GALLERY_LABELS_PATH

    # If debug is on, we shouldn't overwrite the full cache, or we should use a debug cache path.
    # For simplicity in this task, we assume debug runs might overwrite or we just append _debug.
    if debug:
        gallery_emb_path = gallery_emb_path.replace(".npy", "_debug.npy")
        gallery_ids_path = gallery_ids_path.replace(".npy", "_debug.npy")

    gallery_emb, gallery_labels = extract_embeddings(
        model,
        gallery_loader,
        device,
        gallery_emb_path,
        gallery_ids_path,
        load_cached_data=load_cached_data,
    )

    # --- 2. Query Generation (Test Set) ---
    print("Preparing Query...")

    query_emb_path = Config.QUERY_EMB_PATH
    # We don't have a separate path in Config for query IDs (image names), so we define one
    query_ids_path = os.path.join(Config.WORKING_DIR, "query_names.npy")

    if debug:
        query_emb_path = query_emb_path.replace(".npy", "_debug.npy")
        query_ids_path = query_ids_path.replace(".npy", "_debug.npy")

    query_emb, query_names = extract_embeddings(
        model,
        test_loader,
        device,
        query_emb_path,
        query_ids_path,
        load_cached_data=load_cached_data,
    )

    # --- 3. Retrieval (Nearest Neighbors) ---
    print("Running Retrieval (Cosine Similarity)...")

    # Move to GPU for matrix multiplication
    gallery_tensor = torch.from_numpy(gallery_emb).to(device)
    query_tensor = torch.from_numpy(query_emb).to(device)

    # Compute Cosine Similarity: Query (N, D) @ Gallery.T (D, M) -> (N, M)
    # Note: Embeddings are already L2 normalized in extract_embeddings
    sim_matrix = torch.matmul(query_tensor, gallery_tensor.t())

    # Retrieve Top K candidates
    # We retrieve more than 5 initially because we need 5 UNIQUE hotel IDs
    # and multiple gallery images might belong to the same hotel.
    k_retrieval = min(Config.KNN_K, len(gallery_labels))
    print(f"Retrieving top {k_retrieval} candidates...")

    topk_vals, topk_indices = torch.topk(sim_matrix, k=k_retrieval, dim=1)

    topk_indices = topk_indices.cpu().numpy()

    # --- 4. Format Submission ---
    print("Formatting Submission...")

    submission_rows = []

    # Iterate over each query image
    for i, q_name in enumerate(query_names):
        indices = topk_indices[i]

        # Get the hotel label indices for the retrieved gallery images
        retrieved_label_indices = gallery_labels[indices]

        # Aggregate to find top 5 unique hotels
        unique_hotels = []
        seen_labels = set()

        for label_idx in retrieved_label_indices:
            if label_idx not in seen_labels:
                # Convert label index back to original hotel_id
                hotel_id = idx_to_id[label_idx]
                unique_hotels.append(str(hotel_id))
                seen_labels.add(label_idx)

                if len(unique_hotels) == 5:
                    break

        # Fill with placeholders if fewer than 5 found (unlikely given K=50)
        while len(unique_hotels) < 5:
            # Fallback to most frequent class or just duplicate last
            # Ideally shouldn't happen with adequate K and gallery size
            if unique_hotels:
                unique_hotels.append(unique_hotels[-1])
            else:
                unique_hotels.append(str(idx_to_id[0]))  # Fallback

        prediction_str = " ".join(unique_hotels)
        submission_rows.append({"image": q_name, "hotel_id": prediction_str})

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
