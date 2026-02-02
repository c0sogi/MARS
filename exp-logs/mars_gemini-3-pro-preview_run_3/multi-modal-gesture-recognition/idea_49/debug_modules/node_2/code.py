import os
import sys
import shutil
import torch
import numpy as np

# Import library components
from library.config import Config
from library.utils import set_seed, compute_levenshtein, run_length_encoding
from library.data_loader import GestureDataset
from library.model import SRDGN
from library.train import CascadedLoss, train_model
from library.predict import predict


def main():
    print("=== Starting Demo Script ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config paths to use a dedicated demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Override hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.WINDOW_SIZE = 32
    Config.HIDDEN_DIM = 32  # Reduce model size for faster forward pass

    # Clean up previous runs
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Levenshtein Distance
    d1 = compute_levenshtein([1, 2, 3], [1, 2, 3])
    assert d1 == 0, f"Levenshtein error: expected 0, got {d1}"
    d2 = compute_levenshtein([1, 2, 3], [1, 4, 3])  # 1 substitution
    assert d2 == 1, f"Levenshtein error: expected 1, got {d2}"

    # Test Run-Length Encoding
    # 0 is background (separator), min_length=3
    raw_seq = np.array([1, 1, 1, 0, 2, 2, 0, 3, 3, 3, 3])
    # 1s: len 3 -> Keep
    # 2s: len 2 -> Drop
    # 3s: len 4 -> Keep
    rle_res = run_length_encoding(raw_seq, min_length=3)
    assert rle_res == [1, 3], f"RLE error: expected [1, 3], got {rle_res}"

    print("    Utilities verified.")

    # ---------------------------------------------------------
    # 3. Verify Data Loader
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loader...")

    # Load a tiny subset of training data (5 samples)
    # We disable cache loading to force processing logic verification
    train_ds = GestureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        mode="train",
        load_cached_data=False,
        max_samples=5,
    )

    assert len(train_ds) > 0, "Dataset is empty."
    sample = train_ds[0]

    # Check keys
    assert "features" in sample, "Sample missing 'features'."
    assert "labels" in sample, "Sample missing 'labels'."

    # Check shapes
    # features: (WindowSize, 193)
    # labels: (WindowSize,)
    feat_shape = sample["features"].shape
    lbl_shape = sample["labels"].shape

    expected_feat_shape = (Config.WINDOW_SIZE, 193)
    expected_lbl_shape = (Config.WINDOW_SIZE,)

    assert (
        feat_shape == expected_feat_shape
    ), f"Feature shape mismatch: {feat_shape} != {expected_feat_shape}"
    assert (
        lbl_shape == expected_lbl_shape
    ), f"Label shape mismatch: {lbl_shape} != {expected_lbl_shape}"

    print("    Data Loader verified.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cpu")
    model = SRDGN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)

    # Create dummy batch (Batch=2, Time=WindowSize, Dim=193)
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, 193).to(device)

    # Forward pass
    l1, l2, l3 = model(dummy_input)

    # Check output shapes: (Batch, Time, NumClasses)
    expected_out_shape = (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)

    assert l1.shape == expected_out_shape, f"Stage 1 output mismatch: {l1.shape}"
    assert l2.shape == expected_out_shape, f"Stage 2 output mismatch: {l2.shape}"
    assert l3.shape == expected_out_shape, f"Stage 3 output mismatch: {l3.shape}"

    print("    Model architecture verified.")

    # ---------------------------------------------------------
    # 5. Verify Loss Function
    # ---------------------------------------------------------
    print("\n[5] Verifying Loss Function...")

    criterion = CascadedLoss()
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, Config.WINDOW_SIZE)).to(
        device
    )

    loss = criterion([l1, l2, l3], dummy_targets)

    assert isinstance(loss, torch.Tensor), "Loss is not a tensor."
    assert loss.item() > 0, "Loss should be positive."

    print(f"    Loss calculation verified (Value: {loss.item():.4f}).")

    # ---------------------------------------------------------
    # 6. Verify Training Pipeline
    # ---------------------------------------------------------
    print("\n[6] Running Training Pipeline (Integration Test)...")

    # Train on 10 samples for 2 epochs
    # Note: We pass epochs explicitly because the default arg was bound at import time
    train_model(max_samples=10, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    print("    Training pipeline completed.")

    # ---------------------------------------------------------
    # 7. Verify Prediction Pipeline
    # ---------------------------------------------------------
    print("\n[7] Running Prediction Pipeline...")

    # Run inference using the checkpoint generated above
    # This processes the full test set defined in metadata/test.csv (95 samples)
    predict(load_cached_data=False)

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    # Verify submission content length
    with open(sub_path, "r") as f:
        lines = f.readlines()

    # Should correspond to number of test samples (95)
    assert len(lines) == 95, f"Expected 95 predictions, found {len(lines)}"

    print("    Prediction pipeline completed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
