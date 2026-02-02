import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import WhaleDataset, get_transforms, load_and_preprocess_images
from library.model import WhaleModel
from library.engine import WhaleEngine


def generate_predictions(load_cached_data=True, debug=Config.DEBUG):
    """
    Executes the inference pipeline for Whale Species Prediction.

    This function:
    1. Loads the Test data and the Full Known Gallery (Train + Val).
    2. Initializes the model and loads trained weights.
    3. Uses WhaleEngine to generate predictions with Re-ranking and Open-Set rejection.
    4. Saves the results to submission.csv.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed images from .npy cache.
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    # Ensure Config matches the runtime debug flag so Engine uses correct paths internally if needed
    Config.DEBUG = debug

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Starting Inference (Debug={debug})...")

    # ---------------------------------------------------------
    # 1. Prepare Metadata & Label Encoder
    # ---------------------------------------------------------
    # We must reconstruct the label encoder exactly as it was during training.
    # The training logic uses sorted unique IDs from the training set (excluding new_whale).
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_train = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

    unique_ids = sorted(df_train["Id"].unique())
    label_encoder = {label: idx for idx, label in enumerate(unique_ids)}
    num_classes = len(unique_ids)
    print(f"Label Encoder reconstructed. {num_classes} known classes.")

    # ---------------------------------------------------------
    # 2. Construct Enhanced Gallery (Train + Val)
    # ---------------------------------------------------------
    # For inference, we want to match against ALL known whales, so we combine Train and Val.
    df_val = pd.read_csv(Config.VAL_CSV)
    df_val = df_val[df_val["Id"] != "new_whale"].reset_index(drop=True)

    # Handle Debug Slicing
    if debug:
        print(f"DEBUG: Limiting gallery to {Config.DEBUG_SAMPLES} samples per split.")
        df_train = df_train.head(Config.DEBUG_SAMPLES)
        df_val = df_val.head(Config.DEBUG_SAMPLES)

        train_cache = Config.CACHE_TRAIN_IMAGES.replace(".npy", "_debug.npy")
        val_cache = Config.CACHE_VAL_IMAGES.replace(".npy", "_debug.npy")
    else:
        train_cache = Config.CACHE_TRAIN_IMAGES
        val_cache = Config.CACHE_VAL_IMAGES

    print("Loading and processing gallery images...")
    # Load images (handles caching internally)
    train_images = load_and_preprocess_images(df_train, train_cache, load_cached_data)
    val_images = load_and_preprocess_images(df_val, val_cache, load_cached_data)

    # Combine arrays
    gallery_images = np.concatenate([train_images, val_images], axis=0)

    # Map labels to integers
    train_labels = df_train["Id"].map(label_encoder).values.astype(np.int64)
    val_labels = df_val["Id"].map(label_encoder).values.astype(np.int64)
    gallery_labels = np.concatenate([train_labels, val_labels], axis=0)

    # Create Gallery Loader (Validation Transforms - No Augmentation)
    gallery_dataset = WhaleDataset(
        images=gallery_images, labels=gallery_labels, transform=get_transforms("val")
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Gallery ready. Total samples: {len(gallery_dataset)}")

    # ---------------------------------------------------------
    # 3. Construct Test Loader
    # ---------------------------------------------------------
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        df_test = df_test.head(Config.DEBUG_SAMPLES)
        test_cache = Config.CACHE_TEST_IMAGES.replace(".npy", "_debug.npy")
    else:
        test_cache = Config.CACHE_TEST_IMAGES

    print("Loading and processing test images...")
    test_images = load_and_preprocess_images(df_test, test_cache, load_cached_data)

    test_dataset = WhaleDataset(
        images=test_images,
        labels=None,  # No labels for test
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Test set ready. Total samples: {len(test_dataset)}")

    # ---------------------------------------------------------
    # 4. Initialize Model
    # ---------------------------------------------------------
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = WhaleModel(num_classes=num_classes, model_name=Config.MODEL_NAME)

    # ---------------------------------------------------------
    # 5. Execute Prediction Engine
    # ---------------------------------------------------------
    # Initialize engine with dummy train/val loaders (not needed for inference)
    # We pass the label_encoder so it can map predictions back to strings.
    engine = WhaleEngine(
        model=model,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        label_encoder=label_encoder,
    )

    # CRITICAL: Override the engine's default gallery loader (which only contains Train)
    # with our enhanced Combined (Train + Val) loader.
    engine.gallery_loader = gallery_loader

    # Run the prediction pipeline
    # This handles feature extraction, re-ranking, thresholding, and saving to CSV.
    engine.predict_test()

    print("Inference pipeline completed.")
