import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dicom_utils import read_dicom_file
from library.data_processing import (
    get_roi_cache,
    load_patient_volume,
    calculate_roi_index,
)
from library.dataset import BrainTumorDataset, get_datasets
from library.model import AsymmetricGroupedEfficientNet
from library.trainer import Trainer


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override for Demo
    # --------------------------------------------------------------------------
    print("[1/7] Setting up configuration for rapid demonstration...")

    # Ensure reproducibility
    seed_everything(42)

    # Define a working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Override Training Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Setup directories based on new config
    Config.setup_directories()

    # Initialize Logger
    logger = get_logger("DemoRunner")
    logger.info("Configuration configured for demo run.")

    # --------------------------------------------------------------------------
    # 2. Prepare Subset Metadata (to ensure Trainer runs quickly)
    # --------------------------------------------------------------------------
    print("[2/7] Preparing subset metadata...")

    # Create a metadata directory for the demo
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (e.g., 4 train, 2 val, 2 test)
    # We ensure we pick rows where files actually exist (checked by metadata generation, but good to be safe)
    subset_train = orig_train.head(4).copy()
    subset_val = orig_val.head(2).copy()
    subset_test = orig_test.head(2).copy()

    # Save subsets
    demo_train_path = os.path.join(demo_meta_dir, "train.csv")
    demo_val_path = os.path.join(demo_meta_dir, "val.csv")
    demo_test_path = os.path.join(demo_meta_dir, "test.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)
    subset_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA = demo_train_path
    Config.VAL_METADATA = demo_val_path
    Config.TEST_METADATA = demo_test_path

    print(
        f"Subset metadata created: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # --------------------------------------------------------------------------
    # 3. Verify DICOM Reading & ROI Calculation
    # --------------------------------------------------------------------------
    print("[3/7] Verifying DICOM utilities and ROI pipeline...")

    # Pick a sample subject from the subset
    sample_row = subset_train.iloc[0]
    flair_path = os.path.join(Config.INPUT_DIR, sample_row["path_FLAIR"])

    # Test ROI Calculation Logic
    anchor_idx = calculate_roi_index(flair_path)
    print(
        f"Calculated anchor index for subject {sample_row['BraTS21ID']}: {anchor_idx}"
    )
    assert isinstance(anchor_idx, int), "Anchor index must be an integer"
    assert anchor_idx >= 0, "Anchor index cannot be negative"

    # Test ROI Cache Generation
    # We pass the subset dataframe to generate cache only for these subjects
    roi_cache = get_roi_cache(subset_train, load_cached_data=False)
    assert (
        str(sample_row["BraTS21ID"]) in roi_cache
    ), "ROI Cache failed to include subject"

    # Test Volume Loading
    # This uses read_dicom_file internally
    volume = load_patient_volume(sample_row, anchor_idx)

    # Verify Volume Shape: (12 channels, 224 height, 224 width)
    # 12 channels = 4 modalities * 3 slices
    print(f"Loaded volume shape: {volume.shape}")
    expected_shape = (12, 224, 224)
    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Expected {expected_shape}, got {volume.shape}"
    assert volume.dtype == torch.float32, "Volume tensor should be float32"
    assert (
        volume.min() >= 0.0 and volume.max() <= 1.0
    ), "Volume data should be normalized to [0, 1]"

    # --------------------------------------------------------------------------
    # 4. Verify Dataset Class
    # --------------------------------------------------------------------------
    print("[4/7] Verifying BrainTumorDataset...")

    ds = BrainTumorDataset(subset_train, roi_cache, phase="train")
    assert len(ds) == len(subset_train), "Dataset length mismatch"

    # Fetch one item
    img_tensor, label_tensor, subject_id = ds[0]

    assert img_tensor.shape == expected_shape, "Dataset returned incorrect image shape"
    assert isinstance(label_tensor, torch.Tensor), "Label must be a tensor"
    assert label_tensor.numel() == 1, "Label must be a scalar"
    print("Dataset verification passed.")

    # --------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("[5/7] Verifying AsymmetricGroupedEfficientNet...")

    model = AsymmetricGroupedEfficientNet()
    model.eval()

    # Create a dummy batch: (BatchSize=2, Channels=12, H=224, W=224)
    dummy_input = torch.randn(2, 12, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (BatchSize=2, NumClasses=1)
    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model architecture verification passed.")

    # --------------------------------------------------------------------------
    # 6. Run Training Loop (Trainer)
    # --------------------------------------------------------------------------
    print("[6/7] Executing Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Run fit
    # This calls get_datasets internally, which will use our modified Config paths
    # and generate the ROI cache for the subsets.
    trainer.fit(load_cached_data=False)

    # Check if model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint was not saved after training."
    print("Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 7. Run Inference (Prediction)
    # --------------------------------------------------------------------------
    print("[7/7] Executing Inference...")

    trainer.predict(load_cached_data=False)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file rows: {len(df_sub)}")
    assert len(df_sub) == len(
        subset_test
    ), "Submission file row count does not match test set size."
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission columns missing."

    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_demo()
