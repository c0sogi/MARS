import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import read_dicom_robust, resize_image, normalize_image
from library.roi_selection import generate_roi_cache, ROIGenerator
from library.dataset import BrainTumorDataset, create_dataloaders, get_transforms
from library.model import AsymmetricGroupedEfficientNet
from library.engine import run_training, generate_submission


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    print(">>> [1/7] Setting up configuration and environment...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Configure paths for the demo to avoid overwriting real work
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed
    Config.IDEA_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Reduce compute requirements for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    # Define paths for demo metadata
    demo_train_meta_path = os.path.join(DEMO_DIR, "train_demo.csv")
    demo_val_meta_path = os.path.join(DEMO_DIR, "val_demo.csv")
    demo_test_meta_path = os.path.join(DEMO_DIR, "test_demo.csv")

    logger = get_logger("demo")
    logger.info(f"Demo directory: {DEMO_DIR}")

    # --------------------------------------------------------------------------
    # 2. Prepare Demo Data Subset
    # --------------------------------------------------------------------------
    print(">>> [2/7] Preparing data subset...")

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_METADATA)
    full_val = pd.read_csv(Config.VAL_METADATA)
    full_test = pd.read_csv(Config.TEST_METADATA)

    # Sample a small subset (e.g., 8 train, 4 val, 4 test)
    # This ensures the code runs in < 5 minutes
    subset_train = full_train.head(8).copy()
    subset_val = full_val.head(4).copy()
    subset_test = full_test.head(4).copy()

    # Save subset metadata
    subset_train.to_csv(demo_train_meta_path, index=False)
    subset_val.to_csv(demo_val_meta_path, index=False)
    subset_test.to_csv(demo_test_meta_path, index=False)

    logger.info(
        f"Created demo metadata: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # --------------------------------------------------------------------------
    # 3. Test Data Processing Functions
    # --------------------------------------------------------------------------
    print(">>> [3/7] Verifying data processing logic...")

    # Pick a sample file from the subset
    sample_row = subset_train.iloc[0]
    sample_flair_dir = os.path.join(Config.INPUT_DIR, sample_row["path_FLAIR"])

    # Find a DICOM file
    sample_files = [f for f in os.listdir(sample_flair_dir) if f.endswith(".dcm")]
    if sample_files:
        sample_file_path = os.path.join(sample_flair_dir, sample_files[0])

        # Test Read
        img = read_dicom_robust(sample_file_path)
        assert isinstance(
            img, np.ndarray
        ), "read_dicom_robust should return numpy array"
        assert img.dtype == np.float32, "Image should be float32"

        # Test Resize
        target_size = (128, 128)  # Test arbitrary size
        img_resized = resize_image(img, target_size)
        assert (
            img_resized.shape == target_size
        ), f"Resize failed. Expected {target_size}, got {img_resized.shape}"

        # Test Normalize
        img_norm = normalize_image(img_resized)
        assert (
            0.0 <= img_norm.min() and img_norm.max() <= 1.0
        ), "Normalization should be [0, 1]"

        logger.info("Data processing functions verified.")
    else:
        logger.warning("No DICOM files found in sample directory to test processing.")

    # --------------------------------------------------------------------------
    # 4. ROI Selection & Cache Generation
    # --------------------------------------------------------------------------
    print(">>> [4/7] Generating ROI cache for subset...")

    # Combine subsets for cache generation
    combined_df = pd.concat([subset_train, subset_val, subset_test], ignore_index=True)

    # Generate cache
    # This uses the library function which handles logic internally
    roi_cache = generate_roi_cache(combined_df, load_cached_data=False)

    # Verify cache structure
    expected_cols = ["BraTS21ID"]
    # We expect columns like FLAIR_0, FLAIR_1, ..., T2w_2
    for mod in Config.INPUT_MODALITIES:
        for i in range(Config.NUM_SLICES_PER_MODALITY):
            expected_cols.append(f"{mod}_{i}")

    for col in expected_cols:
        assert col in roi_cache.columns, f"Missing column {col} in ROI cache"

    assert len(roi_cache) == len(combined_df), "ROI cache size mismatch"
    logger.info("ROI Cache generated successfully.")

    # --------------------------------------------------------------------------
    # 5. Dataset & DataLoader Initialization
    # --------------------------------------------------------------------------
    print(">>> [5/7] Initializing DataLoaders...")

    # Create DataLoaders using the factory function
    # We pass the paths to our demo metadata files
    dataloaders = create_dataloaders(
        train_metadata_path=demo_train_meta_path,
        val_metadata_path=demo_val_meta_path,
        test_metadata_path=demo_test_meta_path,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,  # Will load the cache we just generated in step 4
    )

    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test" in dataloaders

    # Verify Batch Shape
    # Get one batch from train loader
    inputs, targets = next(iter(dataloaders["train"]))

    # Expected shape: (Batch, Channels, H, W)
    # Channels = 4 modalities * 3 slices = 12
    expected_shape = (Config.BATCH_SIZE, 12, Config.IMG_SIZE[0], Config.IMG_SIZE[1])

    assert (
        inputs.shape == expected_shape
    ), f"Batch shape mismatch. Expected {expected_shape}, got {inputs.shape}"
    assert targets.shape == (Config.BATCH_SIZE,), "Target shape mismatch"

    logger.info(f"DataLoader operational. Input shape: {inputs.shape}")

    # --------------------------------------------------------------------------
    # 6. Model Initialization & Training Demo
    # --------------------------------------------------------------------------
    print(">>> [6/7] Initializing Model and Running Training Loop...")

    device = Config.DEVICE
    model = AsymmetricGroupedEfficientNet()
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 12, 224, 224).to(device)
        output = model(dummy_input)
        assert output.shape == (
            2,
            1,
        ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run Training
    # This uses the engine.run_training function
    run_training(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        epochs=Config.EPOCHS,
        device=device,
        save_path=Config.MODEL_SAVE_PATH,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    logger.info("Training loop completed and model saved.")

    # --------------------------------------------------------------------------
    # 7. Inference & Submission
    # --------------------------------------------------------------------------
    print(">>> [7/7] Running Inference and Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate Submission
    generate_submission(
        model=model,
        test_loader=dataloaders["test"],
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Submission columns mismatch"
    assert len(df_sub) == len(
        subset_test
    ), f"Submission length mismatch. Expected {len(subset_test)}, got {len(df_sub)}"

    logger.info(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
