import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from library.config import Config
from library.dataset import get_dataset
from library.model_factory import load_backbone


def extract_features(stream_config, split, load_cached_data=True, debug_limit=None):
    """
    Extracts, processes, and caches features for a specific stream and split.

    Applies Test Time Augmentation (Horizontal Flip) and Multi-View Early Fusion
    (Global + Standard + Local concatenation).

    Args:
        stream_config (dict): Configuration dictionary for the stream (STREAM_A or STREAM_B).
        split (str): Dataset split ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from disk.
        debug_limit (int, optional): If set, only process this many samples.
                                     Disables loading/saving to main cache.

    Returns:
        tuple: (embeddings, ids, labels)
            embeddings (np.ndarray): Shape (N, Feature_Dim * 3)
            ids (np.ndarray): Shape (N,)
            labels (np.ndarray or None): Shape (N,)
    """
    # Construct cache paths
    # We use 'fused' as the view name for the concatenated embedding file
    emb_path = Config.get_embeddings_path(stream_config, split, "fused")
    ids_path = Config.get_ids_path(stream_config, split)
    lbl_path = Config.get_labels_path(stream_config, split)

    # 1. Try Loading from Cache
    # Only load if requested and NOT in debug mode (debug runs should not rely on full cache)
    if load_cached_data and debug_limit is None:
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            print(f"Loading cached features for {stream_config['name']} ({split})...")
            try:
                embeddings = np.load(emb_path)
                ids = np.load(ids_path)
                # Labels might not exist for test set or if previously not saved
                labels = np.load(lbl_path) if os.path.exists(lbl_path) else None
                return embeddings, ids, labels
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Extracting features for {stream_config['name']} ({split})...")

    # Initialize Model
    model = load_backbone(stream_config)
    # Ensure model is in eval mode (handled by factory, but good practice)
    model.eval()

    # Initialize Dataset
    dataset = get_dataset(split, stream_config)

    # Handle Debugging
    if debug_limit is not None:
        print(f"Debug mode: Processing only {debug_limit} samples.")
        indices = range(min(len(dataset), debug_limit))
        dataset = Subset(dataset, indices)

    # Initialize DataLoader
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
    )

    all_embeddings = []
    all_ids = []
    all_labels = []
    has_labels = False

    device = Config.DEVICE

    # Inference Loop
    with torch.no_grad():
        for batch in loader:
            # Collect Metadata
            all_ids.extend(batch["id"])

            if "label" in batch:
                all_labels.extend(batch["label"])
                has_labels = True

            # Extract Views
            views = batch["views"]
            view_names = ["global", "standard", "local"]
            batch_view_embeddings = []

            for v_name in view_names:
                # Shape: (B, C, H, W)
                imgs = views[v_name].to(device)

                # 1. Forward Pass (Original)
                feats_orig = model(imgs)

                # 2. Forward Pass (Flipped) - TTA
                # Flip along width axis (dim 3)
                imgs_flip = torch.flip(imgs, dims=[3])
                feats_flip = model(imgs_flip)

                # 3. Average
                feats_avg = (feats_orig + feats_flip) / 2.0

                batch_view_embeddings.append(feats_avg)

            # Concatenate views along feature dimension (dim 1)
            # Result shape: (B, Feat_Dim * 3)
            fused_feats = torch.cat(batch_view_embeddings, dim=1)

            # Move to CPU and store
            all_embeddings.append(fused_feats.cpu().numpy())

    # Aggregate results
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_ids = np.array(all_ids)
    final_labels = np.array(all_labels) if has_labels else None

    print(f"Extraction complete. Shape: {final_embeddings.shape}")

    # 3. Save to Cache
    # Do not save if in debug mode to prevent cache corruption
    if debug_limit is None:
        print(f"Saving features to cache directory: {Config.WORKING_DIR}")
        np.save(emb_path, final_embeddings)
        np.save(ids_path, final_ids)
        if final_labels is not None:
            np.save(lbl_path, final_labels)

    return final_embeddings, final_ids, final_labels
