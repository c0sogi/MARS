import os
import shutil
import pandas as pd
import torch
import warnings
import numpy as np

# Import from provided library
from library.config import Config
from library.utils import set_seed, LabelMapper
from library.dataset import get_dataloaders
from library.model import HybridCRNN
from library.trainer import Trainer


def run_demo():
    print("=== Starting Speech Command Recognition Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1/6] Configuring environment...")

    # Override Config for a fast demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Moderate batch size
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to this new working dir
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create small subsets for Validation and Test to speed up the 'validate' and 'predict' phases
    # (Training data generation is hardcoded in dataset.py to balance classes, so we leave that as is,
    # but since we only run 1 epoch, it will be acceptable).
    print("      Creating small validation and test subsets...")

    full_val = pd.read_csv(Config.VAL_CSV)
    full_test = pd.read_csv(Config.TEST_CSV)

    # Sample 100 files for validation and test
    small_val = full_val.sample(n=100, random_state=Config.SEED)
    small_test = full_test.sample(n=100, random_state=Config.SEED)

    val_small_path = os.path.join(Config.WORKING_DIR, "val_small.csv")
    test_small_path = os.path.join(Config.WORKING_DIR, "test_small.csv")

    small_val.to_csv(val_small_path, index=False)
    small_test.to_csv(test_small_path, index=False)

    # Override Config paths to point to these small files
    Config.VAL_CSV = val_small_path
    Config.TEST_CSV = test_small_path

    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying Data Loading...")

    # Generate dataloaders (load_cached_data=False forces generation of the balanced train set)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check lengths
    print(f"      Train Batches: {len(train_loader)}")
    print(f"      Val Batches:   {len(val_loader)}")

    # Fetch one batch to verify shapes
    inputs, targets = next(iter(train_loader))

    print(f"      Input Shape:  {inputs.shape}")  # Expected: (Batch, 1, 128, Time)
    print(f"      Target Shape: {targets.shape}")  # Expected: (Batch,)

    # Assertions
    assert inputs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert inputs.shape[1] == 1, "Channel dimension should be 1 (Mono/Spectrogram)."
    assert (
        inputs.shape[2] == Config.N_MELS
    ), f"Frequency dimension should be {Config.N_MELS}."
    assert not torch.isnan(inputs).any(), "Input tensor contains NaNs."
    assert targets.max() < Config.NUM_CLASSES, "Target labels out of bounds."
    assert targets.min() >= 0, "Target labels cannot be negative."

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HybridCRNN().to(device)

    # Move sample batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"      Output Shape: {outputs.shape}")  # Expected: (Batch, Num_Classes)

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch."
    assert not torch.isnan(outputs).any(), "Model output contains NaNs."

    # -------------------------------------------------------------------------
    # 4. Label Mapping Verification
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying Label Logic...")

    mapper = LabelMapper()

    # Test 1: Target Label
    lbl = "up"
    idx = mapper.to_index(lbl)
    sub_lbl = mapper.map_to_submission(lbl)
    assert sub_lbl == "up", f"Mapping error for target '{lbl}'"

    # Test 2: Auxiliary Label
    lbl = "bird"
    idx_aux = mapper.to_index(lbl)
    sub_lbl_aux = mapper.map_to_submission(lbl)
    assert (
        sub_lbl_aux == "unknown"
    ), f"Mapping error for aux '{lbl}' (expected 'unknown')"

    # Test 3: Silence
    lbl = "silence"
    sub_lbl_sil = mapper.map_to_submission(lbl)
    assert sub_lbl_sil == "silence", f"Mapping error for '{lbl}'"

    print("      Label mapping logic confirmed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5/6] Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    # load_cached_data=True reuses the parquet file generated in step 2
    trainer = Trainer(load_cached_data=True)

    # Run training
    trainer.fit()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.CHECKPOINT_PATH
    ), f"Checkpoint not found at {Config.CHECKPOINT_PATH}"
    print("      Training completed and checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Prediction Verification
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating Submission...")

    trainer.predict()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission Rows: {len(df_sub)}")

    # Should match the size of our small test set (100)
    assert len(df_sub) == 100, f"Expected 100 rows in submission, found {len(df_sub)}."

    # Check columns
    assert "fname" in df_sub.columns, "Missing 'fname' column."
    assert "label" in df_sub.columns, "Missing 'label' column."

    # Check label validity (must be one of the 12 allowed)
    allowed_labels = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "silence",
        "unknown",
    }
    predicted_labels = set(df_sub["label"].unique())
    invalid = predicted_labels - allowed_labels
    if invalid:
        raise AssertionError(f"Submission contains invalid labels: {invalid}")

    print("      Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Filter warnings to keep output clean
    warnings.filterwarnings("ignore")
    run_demo()
