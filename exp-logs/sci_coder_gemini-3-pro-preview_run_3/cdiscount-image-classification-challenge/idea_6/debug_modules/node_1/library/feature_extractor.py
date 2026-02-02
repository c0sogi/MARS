import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    NUM_WORKERS,
    TRAIN_META,
    VAL_META,
    TEST_META,
    TRAIN_BSON,
    TEST_BSON,
    TRAIN_FEATURES,
    TRAIN_LABELS,
    VAL_FEATURES,
    VAL_LABELS,
    TEST_FEATURES,
    TEST_IDS,
    CACHE_DIR,
    EMBEDDING_DIM,
    DEBUG_SIZE,
)
from library.data_utils import RawImageDataset, get_category_encoder
from library.model import ProductFeatureExtractor


def extract_features_to_disk(
    metadata_path,
    bson_path,
    output_feat_path,
    output_label_path=None,
    output_id_path=None,
    load_cached_data=True,
    debug=False,
    split_name="train",
):
    """
    Extracts features from images in BSON files and saves them to disk as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        bson_path (str): Path to the source BSON file.
        output_feat_path (str): Path to save feature numpy array.
        output_label_path (str, optional): Path to save label numpy array.
        output_id_path (str, optional): Path to save ID numpy array.
        load_cached_data (bool): Whether to try loading from cache first.
        debug (bool): If True, process a small subset of data.
        split_name (str): Name of the split for logging.

    Returns:
        tuple: (features, labels, ids) as numpy arrays.
    """
    # 1. Check Cache
    cache_exists = os.path.exists(output_feat_path)
    if output_label_path:
        cache_exists = cache_exists and os.path.exists(output_label_path)
    if output_id_path:
        cache_exists = cache_exists and os.path.exists(output_id_path)

    if load_cached_data and cache_exists and not debug:
        print(f"Loading cached features for {split_name} from {CACHE_DIR}...")
        features = np.load(output_feat_path)
        labels = np.load(output_label_path) if output_label_path else None
        ids = np.load(output_id_path) if output_id_path else None
        return features, labels, ids

    # 2. Setup
    print(f"Starting feature extraction for {split_name}...")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if debug:
        print(f"Debug mode: limiting {split_name} to {DEBUG_SIZE} samples.")
        df = df.head(DEBUG_SIZE)

    # Encoder (only needed if we are saving labels)
    encoder = None
    if output_label_path is not None:
        # We always load/fit encoder based on training metadata via get_category_encoder
        encoder = get_category_encoder(load_cached_data=load_cached_data)

    # Dataset & Loader
    # Note: RawImageDataset handles image loading and preprocessing
    dataset = RawImageDataset(df, bson_path, encoder=encoder)

    # batch_size=1 is crucial here because the number of images per product varies (1-4).
    # The ProductFeatureExtractor expects inputs of shape (N, 3, H, W).
    # DataLoader with batch_size=1 yields (1, N, 3, H, W), which we squeeze to (N, 3, H, W).
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Model
    model = ProductFeatureExtractor().to(DEVICE)
    model.eval()

    # Pre-allocate arrays for efficiency
    num_samples = len(df)
    features_arr = np.zeros((num_samples, EMBEDDING_DIM), dtype=np.float32)

    labels_arr = None
    if output_label_path:
        labels_arr = np.zeros(num_samples, dtype=np.int64)

    ids_arr = None
    if output_id_path:
        ids_arr = np.zeros(num_samples, dtype=np.int64)

    # 3. Extraction Loop
    print(f"Extracting features for {num_samples} records...")
    with torch.no_grad():
        for i, (imgs, label, _id) in enumerate(loader):
            # imgs shape: (1, N, 3, H, W) -> squeeze -> (N, 3, H, W)
            imgs = imgs.squeeze(0).to(DEVICE)

            # Extract features (Model handles aggregation internally)
            # Output shape: (1280,)
            embedding = model(imgs)

            # Store in numpy arrays
            features_arr[i] = embedding.cpu().numpy()

            if labels_arr is not None:
                labels_arr[i] = label.item()

            if ids_arr is not None:
                ids_arr[i] = _id.item()

            # Periodic logging
            if (i + 1) % 50000 == 0:
                print(f"[{split_name}] Processed {i + 1}/{num_samples}")

    # 4. Save to Disk
    print(f"Saving {split_name} features to {CACHE_DIR}...")

    np.save(output_feat_path, features_arr)
    if output_label_path:
        np.save(output_label_path, labels_arr)
    if output_id_path:
        np.save(output_id_path, ids_arr)

    print(f"Completed {split_name} feature extraction.")
    return features_arr, labels_arr, ids_arr


def run_feature_extraction(load_cached_data=True, debug=False):
    """
    Orchestrates the feature extraction for Train, Validation, and Test sets.
    """
    # 1. Train Set
    extract_features_to_disk(
        metadata_path=TRAIN_META,
        bson_path=TRAIN_BSON,
        output_feat_path=TRAIN_FEATURES,
        output_label_path=TRAIN_LABELS,
        load_cached_data=load_cached_data,
        debug=debug,
        split_name="train",
    )

    # 2. Validation Set
    extract_features_to_disk(
        metadata_path=VAL_META,
        bson_path=TRAIN_BSON,
        output_feat_path=VAL_FEATURES,
        output_label_path=VAL_LABELS,
        load_cached_data=load_cached_data,
        debug=debug,
        split_name="val",
    )

    # 3. Test Set
    # Note: Test set has no labels, but we need IDs for submission
    extract_features_to_disk(
        metadata_path=TEST_META,
        bson_path=TEST_BSON,
        output_feat_path=TEST_FEATURES,
        output_id_path=TEST_IDS,
        load_cached_data=load_cached_data,
        debug=debug,
        split_name="test",
    )
