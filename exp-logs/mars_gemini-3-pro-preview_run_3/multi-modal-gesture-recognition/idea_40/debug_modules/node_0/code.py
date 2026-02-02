import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# ==========================================
# 1. Configuration Override for Speed/Demo
# ==========================================
# We import config first and patch it before other modules load these constants.
import library.config

print("[Demo] Patching configuration for rapid execution...")
library.config.DEBUG_DATA_LIMIT = 10  # Process only 10 samples
library.config.NUM_EPOCHS = 2  # Train for only 2 epochs
library.config.BATCH_SIZE = 4  # Small batch size
library.config.HIDDEN_DIM = 64  # Smaller model size for speed
library.config.CACHE_DIR = "./working/demo_cache"
library.config.SUBMISSION_FILE = "./working/demo_submission.csv"

# Ensure clean state for demo
if os.path.exists(library.config.CACHE_DIR):
    shutil.rmtree(library.config.CACHE_DIR)
os.makedirs(library.config.CACHE_DIR, exist_ok=True)

# ==========================================
# 2. Imports (Post-Patching)
# ==========================================
from library.utils import set_seed, decode_predictions, compute_challenge_metric
from library.data_loader import process_dataset, GestureDataset, get_dataloaders
from library.model import RMDKN
from library.trainer import Trainer


def main():
    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Running on device: {device}")

    # ==========================================
    # 3. Data Pipeline Verification
    # ==========================================
    print("\n[Demo] Verifying Data Pipeline...")

    # Process a small subset of training data
    # This uses the DEBUG_DATA_LIMIT set above
    train_samples = process_dataset(
        library.config.TRAIN_METADATA_PATH, "dataset_train_demo", load_cached_data=False
    )

    assert (
        len(train_samples) <= library.config.DEBUG_DATA_LIMIT
    ), f"Expected <= {library.config.DEBUG_DATA_LIMIT} samples, got {len(train_samples)}"

    # Check sample structure
    sample = train_samples[0]
    print(f"  Sample ID: {sample['sample_id']}")
    print(f"  Skeleton Shape: {sample['skeleton'].shape}")
    print(f"  Audio Shape: {sample['audio'].shape}")
    print(f"  Labels Shape: {sample['labels'].shape}")

    assert sample["skeleton"].ndim == 3, "Skeleton data should be (Time, Joints, 3)"
    assert sample["audio"].ndim == 2, "Audio data should be (Time, Features)"

    # Initialize Dataset
    dataset = GestureDataset(train_samples, is_train=True)
    print(f"  Dataset Length (Windows): {len(dataset)}")

    # Check item retrieval
    item = dataset[0]
    features = item["features"]
    labels = item["labels"]

    print(f"  Window Features Shape: {features.shape}")  # Expected: (WindowSize, 193)
    print(f"  Window Labels Shape: {labels.shape}")  # Expected: (WindowSize)

    expected_feat_dim = library.config.SKELETON_JOINTS * 3 * 3 + library.config.N_MFCC
    assert features.shape == (
        library.config.WINDOW_SIZE,
        expected_feat_dim,
    ), f"Mismatch in feature shape. Expected ({library.config.WINDOW_SIZE}, {expected_feat_dim}), got {features.shape}"

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[Demo] Verifying Model Architecture...")

    model = RMDKN().to(device)
    model.eval()

    # Create dummy batch: (Batch, Time, Features)
    dummy_input = torch.randn(2, library.config.WINDOW_SIZE, expected_feat_dim).to(
        device
    )

    with torch.no_grad():
        l1, l2, l3 = model(dummy_input)

    print(f"  Logits 1 Shape: {l1.shape}")
    print(f"  Logits 2 Shape: {l2.shape}")
    print(f"  Logits 3 Shape: {l3.shape}")

    expected_out_shape = (2, library.config.WINDOW_SIZE, library.config.NUM_CLASSES)
    assert l1.shape == expected_out_shape, "Stage 1 output shape mismatch"
    assert l2.shape == expected_out_shape, "Stage 2 output shape mismatch"
    assert l3.shape == expected_out_shape, "Stage 3 output shape mismatch"

    print("  Model forward pass successful.")

    # ==========================================
    # 5. Training Loop Verification
    # ==========================================
    print("\n[Demo] Verifying Training Loop...")

    # Initialize Trainer
    # This will internally call get_dataloaders which respects our patched config
    trainer = Trainer(device=device)

    # Run training
    # Since we set NUM_EPOCHS=2 and DEBUG_DATA_LIMIT=10, this should be very fast
    trainer.run()

    # Check if best model was saved
    best_model_path = os.path.join(library.config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not created!"
    print(f"  Training completed. Model saved to {best_model_path}")

    # ==========================================
    # 6. Inference and Metric Verification
    # ==========================================
    print("\n[Demo] Verifying Inference and Metrics...")

    # Pick a validation sample from the trainer's loaded samples
    if len(trainer.val_samples) > 0:
        val_sample = trainer.val_samples[0]

        # Run inference
        avg_probs = trainer.run_inference_on_sample(val_sample)
        print(f"  Inference Output Shape: {avg_probs.shape}")

        assert (
            avg_probs.shape[0] == val_sample["skeleton"].shape[0]
        ), "Inference output length does not match input sequence length"
        assert (
            avg_probs.shape[1] == library.config.NUM_CLASSES
        ), "Inference output classes mismatch"

        # Decode
        pred_seq = decode_predictions(avg_probs)
        print(f"  Predicted Sequence: {pred_seq}")

        # Compute Metric (Self-check against dummy GT)
        # Let's assume the prediction is perfect for the metric check
        score = compute_challenge_metric([pred_seq], [pred_seq])
        assert score == 0.0, "Metric should be 0.0 for identical sequences"

        # Check against a different sequence
        dummy_gt = [1, 2, 3]
        score_diff = compute_challenge_metric([pred_seq], [dummy_gt])
        print(f"  Metric vs [1, 2, 3]: {score_diff:.4f}")
    else:
        print("  No validation samples available (check DEBUG_DATA_LIMIT vs split).")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("\n[Demo] Generating Submission...")

    trainer.generate_submission()

    assert os.path.exists(library.config.SUBMISSION_FILE), "Submission file not found!"

    # Read first few lines
    with open(library.config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        print(f"  Submission Lines Generated: {len(lines)}")
        if len(lines) > 0:
            print(f"  First line: {lines[0].strip()}")

    print("\n[Demo] All verification steps passed successfully.")


if __name__ == "__main__":
    main()
