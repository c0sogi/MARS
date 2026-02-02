import os
import torch
import numpy as np
from library.config import (
    DEVICE,
    WORKING_DIR,
    get_embedding_path,
    get_ids_path,
    get_labels_path,
)
from library.model_factory import load_backbone
from library.dataset import get_dataloaders


def extract_features(stream_config, split, load_cached_data=True):
    """
    Extracts features for a given stream and split using the Multi-View pipeline.
    Implements caching, TTA (Horizontal Flip), and saves results to disk.

    Args:
        stream_config (dict): Configuration dictionary for the stream.
        split (str): Dataset split ('train', 'val', 'test').
        load_cached_data (bool): Whether to use cached data if available.

    Returns:
        dict: Dictionary containing paths to the saved embeddings, ids, and labels for each view.
    """
    stream_name = stream_config["name"]
    views = list(stream_config["views"].keys())

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Check Cache
    # We need all files for all views to be present to consider it a cache hit
    all_cached = True
    output_paths = {}

    for view in views:
        paths = {
            "embeddings": get_embedding_path(stream_name, split, view),
            "ids": get_ids_path(stream_name, split, view),
            "labels": get_labels_path(stream_name, split, view),
        }
        output_paths[view] = paths

        if not (
            os.path.exists(paths["embeddings"])
            and os.path.exists(paths["ids"])
            and os.path.exists(paths["labels"])
        ):
            all_cached = False

    if load_cached_data and all_cached:
        print(
            f"[{stream_name} | {split}] Cache hit. Loading artifacts from {WORKING_DIR}"
        )
        return output_paths

    # 2. Run Extraction
    print(f"[{stream_name} | {split}] Cache miss. Starting feature extraction...")

    # Load Model
    model = load_backbone(stream_config)
    model.to(DEVICE)
    model.eval()

    # Load Data
    # We get all loaders but only use the one for the requested split
    train_loader, val_loader, test_loader, _ = get_dataloaders(stream_config)

    if split == "train":
        loader = train_loader
    elif split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    else:
        raise ValueError(f"Unknown split: {split}")

    # Initialize storage
    # Structure: cache[view]['embeddings'] = list of arrays
    cache = {v: {"embeddings": [], "ids": [], "labels": []} for v in views}

    with torch.no_grad():
        for batch_idx, (views_dict, targets, ids) in enumerate(loader):
            # targets: tensor (B,)
            # ids: tuple (B,)

            targets_np = targets.numpy()

            for view_name, img_tensor in views_dict.items():
                if view_name not in views:
                    continue

                # Move to device
                img = img_tensor.to(DEVICE)

                # TTA: Horizontal Flip
                img_flip = torch.flip(img, dims=[3])

                # Batch concatenation for efficiency
                # Input: (B, C, H, W) -> (2B, C, H, W)
                input_batch = torch.cat([img, img_flip], dim=0)

                # Forward pass
                # Output: (2B, F)
                features = model(input_batch)

                # Split original and flipped
                B = img.shape[0]
                feat_orig = features[:B]
                feat_flip = features[B:]

                # Average features
                feat_avg = (feat_orig + feat_flip) / 2.0

                # Store results (move to CPU)
                cache[view_name]["embeddings"].append(feat_avg.cpu().numpy())
                cache[view_name]["ids"].extend(ids)
                cache[view_name]["labels"].append(targets_np)

    # 3. Save to Disk
    print(f"[{stream_name} | {split}] Saving artifacts...")

    for view in views:
        # Aggregate
        emb_full = np.concatenate(cache[view]["embeddings"], axis=0)
        lbl_full = np.concatenate(cache[view]["labels"], axis=0)
        ids_full = np.array(cache[view]["ids"])

        # Save
        np.save(output_paths[view]["embeddings"], emb_full)
        np.save(output_paths[view]["ids"], ids_full)
        np.save(output_paths[view]["labels"], lbl_full)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"[{stream_name} | {split}] Extraction complete.")
    return output_paths
