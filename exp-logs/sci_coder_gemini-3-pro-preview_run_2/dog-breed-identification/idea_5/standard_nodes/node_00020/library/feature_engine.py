import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import library.config as config
import library.dataset as dataset_lib
import library.backbones as backbones_lib


def extract_features(
    dataset_key,
    model_name,
    weights_name,
    batch_size=config.BATCH_SIZE,
    load_cached_data=True,
    debug=False,
):
    """
    Extracts features using the specified backbone and multi-view strategy.

    Args:
        dataset_key (str): One of 'train', 'val', or 'test'.
        model_name (str): Name of the model architecture.
        weights_name (str): Name of the pretrained weights.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to load from disk if available.
        debug (bool): If True, runs on a subset of data.

    Returns:
        tuple: (embeddings, labels, ids)
            embeddings (np.ndarray): Shape (N, total_dim)
            labels (np.ndarray): Shape (N,)
            ids (np.ndarray): Shape (N,)
    """

    # 1. Determine Metadata Path
    if dataset_key == "train":
        csv_path = config.TRAIN_CSV
    elif dataset_key == "val":
        csv_path = config.VAL_CSV
    elif dataset_key == "test":
        csv_path = config.TEST_CSV
    else:
        raise ValueError(
            f"Invalid dataset_key: {dataset_key}. Must be 'train', 'val', or 'test'."
        )

    # 2. Define Cache Paths
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    cache_prefix = f"{safe_model_name}_{dataset_key}"
    if debug:
        cache_prefix += "_debug"

    embed_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_embeddings.npy")
    label_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_labels.npy")
    id_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_ids.npy")

    # 3. Check Cache
    if load_cached_data:
        if (
            os.path.exists(embed_path)
            and os.path.exists(label_path)
            and os.path.exists(id_path)
        ):
            print(f"Loading cached features for {dataset_key} ({model_name})...")
            embeddings = np.load(embed_path)
            labels = np.load(label_path)
            ids = np.load(id_path)
            return embeddings, labels, ids
        else:
            print(
                f"Cache miss for {dataset_key} ({model_name}). Starting extraction..."
            )

    # 4. Setup Model
    device = torch.device(config.DEVICE)
    model = backbones_lib.load_feature_extractor(
        model_name=model_name, weights_name=weights_name, device=device, freeze=True
    )

    # 5. Setup DataLoader
    # Shuffle is False for feature extraction to maintain order with IDs
    loader = dataset_lib.get_dataloader(
        csv_path=csv_path,
        model_weights=weights_name,
        batch_size=batch_size,
        shuffle=False,
        debug=debug,
    )

    # 6. Extraction Loop
    all_embeddings = []
    all_labels = []
    all_ids = []

    # Explicit order for view concatenation
    view_order = ["standard", "global", "local"]

    with torch.no_grad():
        for views_dict, batch_labels, batch_ids in loader:

            # List to store embeddings for this batch from all views
            batch_view_embeddings = []

            for view_name in view_order:
                if view_name not in views_dict:
                    continue

                # Get images and move to device
                images = views_dict[view_name].to(device)

                # --- Feature-Level TTA ---
                # 1. Forward pass original
                emb_orig = model(images)

                # 2. Forward pass flipped
                # HFlip on tensor (B, C, H, W) is safe after normalization
                # as long as normalization is channel-wise and spatial structure is preserved.
                images_flipped = F.hflip(images)
                emb_flip = model(images_flipped)

                # 3. Average features
                emb_avg = (emb_orig + emb_flip) / 2.0

                batch_view_embeddings.append(emb_avg)

            # --- Early Fusion of Views ---
            # Concatenate embeddings from [Standard, Global, Local] along feature dimension (dim=1)
            # Result shape: (Batch, Dim_Standard + Dim_Global + Dim_Local)
            batch_final_features = torch.cat(batch_view_embeddings, dim=1)

            # Store results (move to CPU)
            all_embeddings.append(batch_final_features.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_ids.extend(batch_ids)

    # 7. Aggregate Results
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.array(all_labels)
    final_ids = np.array(all_ids)

    # 8. Save to Cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    np.save(embed_path, final_embeddings)
    np.save(label_path, final_labels)
    np.save(id_path, final_ids)

    print(f"Features saved to {config.WORKING_DIR}")
    print(f"  Embeddings shape: {final_embeddings.shape}")

    return final_embeddings, final_labels, final_ids
