import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library components
from library.utils import (
    seed_everything,
    read_dicom_robust,
    calculate_roi_index,
    generate_roi_cache,
)
from library.dataset import get_dataloader
from library.model import AsymmetricEfficientNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print(">>> Initializing Demonstration...")
    seed_everything(42)

    # Define paths
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"

    # Clean/Create working directory
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Preparation (Subsetting for Speed)
    # --------------------------------------------------------------------------
    print("\n>>> 1. Preparing Data Subsets...")

    # Load full metadata
    train_full = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_full = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_full = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Create tiny subsets (e.g., 8 train, 4 val, 4 test) to run quickly
    train_subset = train_full.head(8).copy()
    val_subset = val_full.head(4).copy()
    test_subset = test_full.head(4).copy()

    print(f"    Train Subset: {len(train_subset)} samples")
    print(f"    Val Subset:   {len(val_subset)} samples")
    print(f"    Test Subset:  {len(test_subset)} samples")

    # --------------------------------------------------------------------------
    # 3. Testing Library Utilities
    # --------------------------------------------------------------------------
    print("\n>>> 2. Verifying Library Utilities...")

    # A. Test DICOM Reading
    # Get a valid path from the first training sample
    sample_row = train_subset.iloc[0]
    flair_rel_path = sample_row["path_FLAIR"]
    flair_full_path = os.path.join(INPUT_ROOT, flair_rel_path)

    # Find the first DICOM file in that directory
    dcm_files = [f for f in os.listdir(flair_full_path) if f.endswith(".dcm")]
    if dcm_files:
        sample_dcm = os.path.join(flair_full_path, dcm_files[0])
        img = read_dicom_robust(sample_dcm)

        # Assertions
        assert isinstance(
            img, np.ndarray
        ), "read_dicom_robust should return numpy array"
        assert img.shape == (224, 224), f"Expected shape (224, 224), got {img.shape}"
        assert img.dtype == np.float32, f"Expected dtype float32, got {img.dtype}"
        print("    [Pass] read_dicom_robust")
    else:
        print("    [Skip] read_dicom_robust (no dcm files found in sample dir)")

    # B. Test ROI Calculation
    roi_idx = calculate_roi_index(flair_full_path)
    assert isinstance(roi_idx, int), "ROI index must be an integer"
    assert roi_idx >= 0, "ROI index must be non-negative"
    print(f"    [Pass] calculate_roi_index (Calculated index: {roi_idx})")

    # C. Test ROI Cache Generation
    # We generate cache for our subset
    roi_cache = generate_roi_cache(
        train_subset,
        load_cached_data=False,
        cache_dir=WORKING_DIR,
        input_root=INPUT_ROOT,
    )
    assert len(roi_cache) == len(train_subset), "ROI cache size mismatch"
    print("    [Pass] generate_roi_cache")

    # --------------------------------------------------------------------------
    # 4. Testing Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("\n>>> 3. Verifying Dataset & DataLoader...")

    # Create DataLoader using the factory function
    # num_workers=0 for simple sequential debugging
    train_loader = get_dataloader(
        train_subset,
        phase="train",
        batch_size=4,
        num_workers=0,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Verify Shapes
    # Expected: (Batch, 12, 224, 224) -> 12 channels = 4 modalities * 3 slices
    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    assert images.shape == (4, 12, 224, 224), "Incorrect image batch shape"
    assert targets.shape == (4, 1), "Incorrect target batch shape"
    assert images.dtype == torch.float32, "Images must be float32"
    print("    [Pass] DataLoader batch generation")

    # --------------------------------------------------------------------------
    # 5. Testing Model Architecture
    # --------------------------------------------------------------------------
    print("\n>>> 4. Verifying Model Architecture...")

    # Instantiate model (without pretrained weights for speed in this check)
    model = AsymmetricEfficientNet(num_classes=1, pretrained=False)
    model.eval()

    # Forward pass on CPU
    with torch.no_grad():
        outputs = model(images)

    print(f"    Model Output Shape: {outputs.shape}")
    assert outputs.shape == (4, 1), "Model output shape mismatch"
    print("    [Pass] AsymmetricEfficientNet forward pass")

    # --------------------------------------------------------------------------
    # 6. Testing Trainer (Training Loop & Prediction)
    # --------------------------------------------------------------------------
    print("\n>>> 5. Running Training & Inference Loop...")

    # Setup DataLoaders for Validation and Test
    val_loader = get_dataloader(
        val_subset,
        phase="valid",
        batch_size=4,
        num_workers=0,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )
    test_loader = get_dataloader(
        test_subset,
        phase="test",
        batch_size=4,
        num_workers=0,
        input_root=INPUT_ROOT,
        cache_dir=WORKING_DIR,
    )

    # Configure Trainer
    config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint_dir": WORKING_DIR,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    trainer = Trainer(device=device, config=config)

    # A. Run Training (Fit)
    print("    Starting training (2 epochs)...")
    best_auc = trainer.fit(train_loader, val_loader, epochs=2, patience=2)
    print(f"    Training complete. Best AUC: {best_auc:.4f}")

    # Verify checkpoint creation
    expected_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(expected_model_path), "Model checkpoint not saved"
    print("    [Pass] Model training and checkpointing")

    # B. Run Inference (Predict)
    print("    Starting inference on test subset...")
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    trainer.predict(test_loader, output_path=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission rows: {len(sub_df)}")

    # Check format
    assert len(sub_df) == len(test_subset), "Submission row count mismatch"
    assert "BraTS21ID" in sub_df.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in sub_df.columns, "Missing MGMT_value column"
    assert sub_df["MGMT_value"].dtype == np.float64, "Prediction column should be float"

    print("    [Pass] Inference and submission generation")
    print("\n>>> DEMONSTRATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_demonstration()
