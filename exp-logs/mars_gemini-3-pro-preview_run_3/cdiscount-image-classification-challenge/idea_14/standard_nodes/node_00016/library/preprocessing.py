import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.configuration import Config
from library.utilities import seed_everything
from library.dataset import get_extraction_dataloader
from library.architecture import DualBackboneExtractor


def extract_features(
    split: str = "train", load_cached_data: bool = True, debug: bool = False
):
    """
    Extracts features from the raw BSON dataset using the DualBackboneExtractor.
    Aggregates features per product and saves them to disk.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        debug (bool): If True, processes a small subset of the data.

    Returns:
        Tuple of (features, auxiliary_data).
        - features: Numpy array of shape (N, 3328).
        - auxiliary_data: Labels (train/val) or IDs (test).
    """
    seed_everything()

    # 1. Determine Output Paths
    if split == "train":
        feat_path = Config.TRAIN_FEATURES_PATH
        aux_path = Config.TRAIN_LABELS_PATH
    elif split == "val":
        feat_path = Config.VAL_FEATURES_PATH
        aux_path = Config.VAL_LABELS_PATH
    elif split == "test":
        feat_path = Config.TEST_FEATURES_PATH
        aux_path = Config.TEST_IDS_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Caching Logic
    if load_cached_data:
        if os.path.exists(feat_path) and os.path.exists(aux_path):
            print(f"Loading cached features for {split} from {feat_path}...")
            features = np.load(feat_path)
            aux_data = np.load(aux_path)
            return features, aux_data
        else:
            print(f"Cache not found for {split}. Starting extraction...")
    else:
        print(f"Forcing feature extraction for {split}...")

    # 3. Setup Model and DataLoader
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DualBackboneExtractor()
    model.to(device)
    model.eval()

    # Initialize DataLoader
    dataloader = get_extraction_dataloader(split=split, debug=debug)

    # Storage Lists
    all_features = []
    all_aux = []  # Labels or IDs

    # 4. Extraction Loop
    print(f"Extracting features for {split} set (Debug={debug})...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch from bson_collate_fn
            # batch_images: (Sum_K, C, H, W)
            # ids_list: List of product IDs
            # batch_sizes: Tensor (B,)
            # batch_l3_indices: Tensor (B,)
            batch_images, ids_list, batch_sizes, batch_l3_indices = batch_data

            # Move to device
            batch_images = batch_images.to(device)
            batch_sizes = batch_sizes.to(device)

            # Forward Pass (includes aggregation)
            # Output: (B, 3328)
            features = model(batch_images, batch_sizes=batch_sizes)

            # Move to CPU and store
            all_features.append(features.cpu().numpy())

            # Store auxiliary data
            if split in ["train", "val"]:
                # Store L3 labels
                all_aux.append(batch_l3_indices.numpy())
            else:
                # Store Product IDs
                all_aux.append(np.array(ids_list, dtype=np.int64))

            # Optional: Clear cache to manage VRAM
            # torch.cuda.empty_cache()

    # 5. Concatenate and Save
    if len(all_features) > 0:
        final_features = np.concatenate(all_features, axis=0)
        final_aux = np.concatenate(all_aux, axis=0)
    else:
        # Handle empty case
        final_features = np.empty((0, Config.TOTAL_FEATURE_DIM), dtype=np.float32)
        final_aux = np.empty((0,), dtype=np.int64)

    print(f"Saving {final_features.shape[0]} records to {Config.WORKING_DIR}...")

    np.save(feat_path, final_features)
    np.save(aux_path, final_aux)

    print(f"Feature extraction for {split} complete.")

    return final_features, final_aux
