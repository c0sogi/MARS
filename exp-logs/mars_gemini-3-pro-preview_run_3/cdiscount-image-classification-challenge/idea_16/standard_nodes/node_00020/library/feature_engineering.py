import os
import torch
import numpy as np
from library.config import Config
from library.data import get_extraction_loader
from library.models import DualBackboneExtractor


def extract_features_for_split(
    split_name,
    meta_path,
    bson_path,
    feat_save_path,
    target_save_path,
    is_test=False,
    subset_size=None,
):
    """
    Runs the feature extraction model on a specific data split and saves the results to disk.
    """
    print(f"Starting feature extraction for {split_name} split...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)

    # Initialize Model
    device = Config.DEVICE
    print(f"Initializing DualBackboneExtractor on {device}...")
    model = DualBackboneExtractor().to(device)
    model.eval()

    # Initialize Loader
    print(f"Creating DataLoader for {split_name}...")
    loader = get_extraction_loader(
        metadata_path=meta_path, bson_path=bson_path, subset_size=subset_size
    )

    # Storage for features and targets/ids
    all_features = []
    all_targets = []

    # Inference Loop
    total_batches = len(loader)
    print_interval = max(1, total_batches // 10)

    print(f"Processing {total_batches} batches...")

    with torch.no_grad():
        for batch_idx, (images, sizes, product_ids, category_ids) in enumerate(loader):
            images = images.to(device)
            sizes = sizes.to(device)

            # Forward pass
            # Model handles mean pooling based on sizes to return (Batch_Size, 3328)
            features = model(images, sizes=sizes)

            # Move to CPU and numpy immediately to save GPU memory
            features_np = features.cpu().numpy().astype(np.float32)
            all_features.append(features_np)

            if is_test:
                # For test, we save product_ids to map predictions later
                targets_np = product_ids.numpy().astype(np.int64)
            else:
                # For train/val, we save category_ids for training labels
                targets_np = category_ids.numpy().astype(np.int64)

            all_targets.append(targets_np)

            if (batch_idx + 1) % print_interval == 0:
                print(f"[{split_name}] Processed batch {batch_idx + 1}/{total_batches}")

    # Concatenate all batches
    print(f"Concatenating {len(all_features)} batches...")
    final_features = np.concatenate(all_features, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)

    # Save to disk
    print(f"Saving features to {feat_save_path}")
    np.save(feat_save_path, final_features)

    print(f"Saving targets/ids to {target_save_path}")
    np.save(target_save_path, final_targets)

    print(f"Completed {split_name}. Output shape: {final_features.shape}")

    # Cleanup to free resources for next split
    del model
    del loader
    del all_features
    del all_targets
    del final_features
    del final_targets
    torch.cuda.empty_cache()


def extract_and_cache_features(load_cached_data=True, subset_size=None):
    """
    Main entry point for the feature engineering pipeline.
    Iterates through Train, Validation, and Test splits.
    Checks for existing cached files to avoid redundant computation.

    Args:
        load_cached_data (bool): If True, skips processing if output files exist.
        subset_size (int, optional): If provided, processes only a subset of data (for debugging).
    """

    # Define the processing tasks based on Config paths
    # Note: Validation split uses the same source BSON as Train (train.bson)
    tasks = [
        {
            "name": "Train",
            "meta": Config.TRAIN_META_PATH,
            "bson": Config.TRAIN_BSON_PATH,
            "feat": Config.TRAIN_FEATURES_PATH,
            "target": Config.TRAIN_LABELS_PATH,
            "is_test": False,
        },
        {
            "name": "Validation",
            "meta": Config.VAL_META_PATH,
            "bson": Config.TRAIN_BSON_PATH,
            "feat": Config.VAL_FEATURES_PATH,
            "target": Config.VAL_LABELS_PATH,
            "is_test": False,
        },
        {
            "name": "Test",
            "meta": Config.TEST_META_PATH,
            "bson": Config.TEST_BSON_PATH,
            "feat": Config.TEST_FEATURES_PATH,
            "target": Config.TEST_IDS_PATH,
            "is_test": True,
        },
    ]

    # Global cache check
    if load_cached_data:
        all_exist = True
        for task in tasks:
            if not (os.path.exists(task["feat"]) and os.path.exists(task["target"])):
                all_exist = False
                break

        if all_exist:
            print("All feature files found in cache. Skipping extraction.")
            return

    print("Starting full feature extraction pipeline...")

    for task in tasks:
        # Individual task cache check
        if (
            load_cached_data
            and os.path.exists(task["feat"])
            and os.path.exists(task["target"])
        ):
            print(f"Cache found for {task['name']}, skipping...")
            continue

        extract_features_for_split(
            split_name=task["name"],
            meta_path=task["meta"],
            bson_path=task["bson"],
            feat_save_path=task["feat"],
            target_save_path=task["target"],
            is_test=task["is_test"],
            subset_size=subset_size,
        )

    print("Feature engineering pipeline finished successfully.")
