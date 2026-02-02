import os
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

from library.config import (
    DEVICE,
    MODEL_STATE_DICT_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGETS_PATH,
    VAL_FEATURES_PATH,
    VAL_TARGETS_PATH,
    TEST_FEATURES_PATH,
    WORKING_DIR,
)
from library.dataset import get_dataloaders
from library.model import SiameseNetwork


def extract_and_save(model, loader, feature_path, target_path=None, desc="Extracting"):
    """
    Helper function to iterate over a data loader, extract features using the model,
    and save the results to disk.
    """
    model.eval()
    all_features = []
    all_targets = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Iterate through batches
        # We use a simple loop without tqdm if verbose is not desired,
        # but for long processes it's helpful.
        # Per instructions "Only print the required information", we'll keep it simple.
        for batch in loader:
            # Move inputs to device
            q_input_ids = batch["q_input_ids"].to(DEVICE)
            q_attention_mask = batch["q_attention_mask"].to(DEVICE)
            a_input_ids = batch["a_input_ids"].to(DEVICE)
            a_attention_mask = batch["a_attention_mask"].to(DEVICE)

            # Extract features (returns [u, v, |u-v|, u*v])
            with autocast():
                features = model.extract_features(
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask
                )

            # Move to CPU and numpy
            all_features.append(features.cpu().float().numpy())

            # Handle targets if they exist and are requested
            if target_path is not None and "labels" in batch:
                all_targets.append(batch["labels"].cpu().numpy())

    # Concatenate all batches
    if len(all_features) > 0:
        features_array = np.vstack(all_features)
        print(f"Saving features to {feature_path}. Shape: {features_array.shape}")
        np.save(feature_path, features_array)
    else:
        print(f"Warning: No features extracted for {desc}")

    if target_path is not None:
        if len(all_targets) > 0:
            targets_array = np.vstack(all_targets)
            print(f"Saving targets to {target_path}. Shape: {targets_array.shape}")
            np.save(target_path, targets_array)
        else:
            print(f"Warning: No targets extracted for {desc}")


def cache_features(debug=False, load_cached_data=True):
    """
    Main function to extract features from the fine-tuned model and save them to disk.
    Implements caching logic to skip extraction if files already exist.

    Args:
        debug (bool): If True, uses a subset of data.
        load_cached_data (bool): If True, checks for existing .npy files before processing.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define the expected artifacts
    artifacts = [
        TRAIN_FEATURES_PATH,
        TRAIN_TARGETS_PATH,
        VAL_FEATURES_PATH,
        VAL_TARGETS_PATH,
        TEST_FEATURES_PATH,
    ]

    # 1. Check Cache
    if load_cached_data:
        missing_artifacts = [p for p in artifacts if not os.path.exists(p)]
        if not missing_artifacts:
            print("All cached feature files exist. Skipping feature extraction.")
            return
        else:
            print(
                f"Cached files missing ({len(missing_artifacts)}). Starting feature extraction..."
            )
    else:
        print("Force reloading requested. Starting feature extraction...")

    # 2. Load Model
    if not os.path.exists(MODEL_STATE_DICT_PATH):
        raise FileNotFoundError(
            f"Model state dict not found at {MODEL_STATE_DICT_PATH}. "
            "Please run Stage 1 training first."
        )

    print(f"Loading model architecture and weights from {MODEL_STATE_DICT_PATH}...")
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=DEVICE))

    # 3. Get DataLoaders
    # Note: We pass load_cached_data to get_dataloaders to potentially reuse the processed text parquet files
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 4. Extract and Save
    print("Extracting features for Training set...")
    extract_and_save(
        model,
        train_loader,
        TRAIN_FEATURES_PATH,
        target_path=TRAIN_TARGETS_PATH,
        desc="Train",
    )

    print("Extracting features for Validation set...")
    extract_and_save(
        model,
        val_loader,
        VAL_FEATURES_PATH,
        target_path=VAL_TARGETS_PATH,
        desc="Validation",
    )

    print("Extracting features for Test set...")
    extract_and_save(
        model, test_loader, TEST_FEATURES_PATH, target_path=None, desc="Test"
    )

    print("Feature caching completed successfully.")
