import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import dicom_processing
from library import dataset
from library import model as lib_model
from library import train_eval
from library import inference


def run_demo():
    print("=== Starting Glioblastoma Subtype Prediction Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1/6] Configuring environment for fast demonstration...")

    # Override configuration for speed and isolation
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 4  # Small batch size
    config.DEBUG = True  # Enable debug mode (subsets data)
    config.MAX_DEBUG_SAMPLES = 10  # Use only 10 samples for training/testing
    config.WORKING_DIR = "./working/demo_run"  # Separate working dir

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    utils.seed_everything(seed=42)

    # Initialize logger
    logger = utils.get_logger(
        "demo", log_file=os.path.join(config.WORKING_DIR, "demo.log")
    )
    logger.info("Configuration complete. Running in DEBUG mode.")

    # -------------------------------------------------------------------------
    # 2. Verify DICOM Processing Logic
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying DICOM processing functions...")

    # Load training metadata to find a real file path
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    if len(df_train) == 0:
        raise ValueError("Training metadata is empty.")

    # Pick the first subject
    sample_row = df_train.iloc[0]
    flair_rel_path = sample_row["path_FLAIR"]
    flair_full_path = os.path.join(config.INPUT_DIR, flair_rel_path)

    # Get list of DICOM files
    dcm_files = [
        os.path.join(flair_full_path, f)
        for f in os.listdir(flair_full_path)
        if f.endswith(".dcm")
    ]

    if dcm_files:
        sample_file = dcm_files[0]

        # Test 1: Read DICOM
        img_raw = dicom_processing.read_dicom_robust(sample_file)
        if not isinstance(img_raw, np.ndarray):
            raise AssertionError("read_dicom_robust did not return a numpy array.")

        # Test 2: Preprocess
        img_proc = dicom_processing.preprocess_slice(img_raw)
        expected_shape = (config.IMG_SIZE, config.IMG_SIZE)
        if img_proc.shape != expected_shape:
            raise AssertionError(
                f"Preprocess shape mismatch. Got {img_proc.shape}, expected {expected_shape}"
            )
        if img_proc.dtype != np.float32:
            raise AssertionError(
                f"Preprocess dtype mismatch. Got {img_proc.dtype}, expected float32"
            )
        if img_proc.max() > 1.0 or img_proc.min() < 0.0:
            raise AssertionError("Image normalization failed (values outside [0, 1]).")

        # Test 3: ROI Selection (Anchor)
        anchor_idx = dicom_processing.select_roi_indices(dcm_files)
        if not isinstance(anchor_idx, int):
            raise AssertionError("ROI anchor index is not an integer.")

        # Test 4: Stride Indices
        indices = dicom_processing.get_dual_stride_indices(anchor_idx, len(dcm_files))
        if len(indices) != 6:
            raise AssertionError(
                f"Stride indices length mismatch. Got {len(indices)}, expected 6."
            )

        logger.info("DICOM processing verification passed.")
    else:
        logger.warning(
            "No DICOM files found in sample directory. Skipping low-level verification."
        )

    # -------------------------------------------------------------------------
    # 3. Verify Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Dataset and DataLoader...")

    # Instantiate Training Dataset
    train_ds = dataset.RSNADataset(
        split="train", transform=dataset.get_transforms("train"), debug=True
    )

    # Check length
    if len(train_ds) > config.MAX_DEBUG_SAMPLES:
        raise AssertionError("Dataset did not respect MAX_DEBUG_SAMPLES limit.")

    # Fetch a single item
    img_tensor, target_tensor = train_ds[0]

    # Verify Input Shape: [24, 224, 224]
    # 4 modalities * 3 slices * 2 strides = 24 channels
    expected_tensor_shape = (config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    if img_tensor.shape != expected_tensor_shape:
        raise AssertionError(
            f"Dataset tensor shape mismatch. Got {img_tensor.shape}, expected {expected_tensor_shape}"
        )

    # Verify Target
    if not isinstance(target_tensor, torch.Tensor):
        raise AssertionError("Target is not a torch.Tensor.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_ds = dataset.RSNADataset(
        split="val", transform=dataset.get_transforms("valid"), debug=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    logger.info(
        f"Dataset verified. Train size: {len(train_ds)}, Val size: {len(val_ds)}"
    )

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying Model Architecture...")

    net = lib_model.AsymmetricEfficientNet()
    net.eval()

    # Create dummy input batch
    dummy_input = torch.randn(
        2, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE
    )

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    # Check output shape: [Batch_Size, 1]
    if output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Got {output.shape}, expected (2, 1)."
        )

    logger.info("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/6] Executing Training Loop (1 Epoch)...")

    # We use the provided run_training function
    # This handles optimizer creation, training, validation, and checkpointing
    trained_model = train_eval.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
        patience=1,
    )

    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        # In a 1-epoch run, if validation fails or doesn't improve (unlikely with init 0),
        # it might not save. However, the logic usually saves at least once if val_auc > 0.
        # If it didn't save, we manually save for the inference step.
        logger.info(
            "Best model not found (likely due to short run). Saving current state."
        )
        utils.save_checkpoint(
            {"state_dict": trained_model.state_dict(), "epoch": 1}, best_model_path
        )

    logger.info("Training loop execution complete.")

    # -------------------------------------------------------------------------
    # 6. Run Inference & Generate Submission
    # -------------------------------------------------------------------------
    print("\n[6/6] Running Inference and Generating Submission...")

    submission_filename = "demo_submission.csv"

    # Run inference pipeline
    inference.generate_submission(
        model_weights_path=best_model_path,
        output_file=submission_filename,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        debug=True,
    )

    # Verify output file
    submission_path = os.path.join("./submission", submission_filename)
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    # Verify content format
    df_sub = pd.read_csv(submission_path)
    required_cols = ["BraTS21ID", "MGMT_value"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(
            f"Submission file missing required columns: {required_cols}"
        )

    if len(df_sub) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"\nSUCCESS: Demo completed successfully.")
    print(f"Submission saved to: {submission_path}")
    print(df_sub.head())


if __name__ == "__main__":
    run_demo()
