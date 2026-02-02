import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import CACHE_DIR, DEVICE, NUM_WORKERS
from library.data_loader import product_collate_fn


def extract_features(
    dataset,
    model,
    batch_size,
    device=DEVICE,
    num_workers=NUM_WORKERS,
    cache_prefix="train",
    load_cached_data=True,
):
    """
    Iterates over the dataset using the frozen model to extract and aggregate features.

    Args:
        dataset (Dataset): The ProductImageDataset.
        model (nn.Module): The FrozenResNet model.
        batch_size (int): Batch size for inference.
        device (str): Device to run on.
        num_workers (int): Number of dataloader workers.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (embeddings, labels, product_ids) as numpy arrays.
    """

    # Define cache file paths
    emb_path = os.path.join(CACHE_DIR, f"{cache_prefix}_embeddings.npy")
    lbl_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")
    ids_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(emb_path)
            and os.path.exists(lbl_path)
            and os.path.exists(ids_path)
        ):
            print(f"Loading cached features for '{cache_prefix}' from {CACHE_DIR}...")
            try:
                embeddings = np.load(emb_path)
                labels = np.load(lbl_path)
                ids = np.load(ids_path)
                return embeddings, labels, ids
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(
                f"Cache not found for '{cache_prefix}'. Starting feature extraction..."
            )
    else:
        print(f"Forcing feature extraction for '{cache_prefix}'...")

    # 2. Setup Data Loader
    # pin_memory=True speeds up transfer to GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=product_collate_fn,
        pin_memory=True,
    )

    # 3. Setup Model
    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_ids = []

    # 4. Inference Loop
    print(f"Starting inference on {len(dataset)} products...")
    with torch.no_grad():
        for i, (flat_imgs, counts, lbls, pids) in enumerate(loader):
            # Move inputs to device
            flat_imgs = flat_imgs.to(device)
            counts = counts.to(device)

            # Forward Pass: Extract features for all images in batch
            # flat_imgs shape: [Total_Images_In_Batch, 3, 224, 224]
            # features shape: [Total_Images_In_Batch, 512]
            features = model(flat_imgs)

            # Aggregate Features per Product (Mean Pooling)
            # We use index_add_ for efficient GPU aggregation

            # Create indices mapping each image to its product index in the batch
            # e.g., counts=[2, 1] -> indices=[0, 0, 1]
            batch_size_curr = len(counts)
            indices = torch.repeat_interleave(
                torch.arange(batch_size_curr, device=device), counts
            )

            # Initialize container for aggregated features
            # Shape: [Batch_Size, 512]
            sum_features = torch.zeros(batch_size_curr, features.size(1), device=device)

            # Sum features based on indices
            sum_features.index_add_(0, indices, features)

            # Divide by counts to get mean
            # counts shape: [Batch_Size] -> [Batch_Size, 1] for broadcasting
            avg_features = sum_features / counts.unsqueeze(1).float()

            # Store results (move to CPU to save GPU memory)
            all_embeddings.append(avg_features.cpu().numpy())
            all_labels.append(lbls.numpy())
            all_ids.append(pids.numpy())

            # Optional: Print status occasionally to indicate aliveness
            if (i + 1) % 500 == 0:
                print(f"Processed batch {i + 1}/{len(loader)}")

    # 5. Concatenate and Save
    print("Concatenating results...")
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    print(f"Saving features to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    np.save(ids_path, ids)

    print("Feature extraction complete.")
    return embeddings, labels, ids
