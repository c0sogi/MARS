import os
import torch
import numpy as np
import shutil
from library.config import Config, set_seed
from library.data_loader import GestureDataset, get_dataloaders
from library.model import HCRGCN
from library.loss import CombinedLoss
from library.trainer import Trainer
from library.utils import (
    decode_predictions,
    compute_levenshtein,
    median_filter_predictions,
)


def main():
    # 1. Setup and Configuration Override
    print("--- Setting up Configuration and Seeding ---")
    set_seed(42)

    # Override Config for a quick demo run
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_run"

    # Update dependent paths manually since Config logic runs at import time
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Create directories
    for d in [
        Config.WORKING_DIR,
        Config.CACHE_DIR,
        Config.CHECKPOINT_DIR,
        Config.SUBMISSION_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n--- Testing Data Loading ---")
    # Initialize Dataset (this will trigger processing and caching if not present)
    # We use the validation set for demonstration to ensure we have labels
    val_dataset = GestureDataset(Config.VAL_METADATA, split="val")

    print(f"Dataset size: {len(val_dataset)}")
    assert len(val_dataset) > 0, "Dataset should not be empty"

    # Get a single item
    features, labels, boundaries = val_dataset[0]
    print(
        f"Single Item Shapes -> Features: {features.shape}, Labels: {labels.shape}, Boundaries: {boundaries.shape}"
    )

    # Verify Feature Dimensions (Skeleton + Velocity + Audio)
    # 36 (Pos) + 36 (Vel) + 13 (Audio) = 85
    assert features.shape[1] == 85, f"Expected feature dim 85, got {features.shape[1]}"

    # Test DataLoader
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    b_features, b_labels, b_boundaries, b_mask, b_lengths = batch

    print(f"Batch Shapes -> Features: {b_features.shape}, Mask: {b_mask.shape}")
    assert b_features.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert b_features.shape[2] == 85, "Feature dimension mismatch in batch"

    # 3. Model Demonstration
    print("\n--- Testing Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCRGCN().to(device)

    # Move batch to device
    b_features = b_features.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)
    b_boundaries = b_boundaries.to(device)

    # Forward Pass
    predictions = model(b_features, b_mask)

    # Verify Output Structure
    assert "stage1" in predictions
    assert "stage2" in predictions
    assert "stage3" in predictions

    s3_cls, s3_bnd = predictions["stage3"]
    print(
        f"Stage 3 Output -> Class Probs: {s3_cls.shape}, Boundary Probs: {s3_bnd.shape}"
    )

    # Check dimensions: (Batch, Time, NumClasses)
    # NumClasses is 21 (20 gestures + 1 background)
    assert s3_cls.shape[2] == 21, f"Expected 21 classes, got {s3_cls.shape[2]}"
    assert s3_bnd.shape[2] == 1, "Expected 1 boundary channel"

    # 4. Loss Calculation Demonstration
    print("\n--- Testing Loss Function ---")
    criterion = CombinedLoss().to(device)
    loss = criterion(predictions, b_labels, b_boundaries, b_mask)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"

    # 5. Trainer Demonstration (Training Loop)
    print("\n--- Testing Trainer (Fit & Predict) ---")
    trainer = Trainer(device=device)

    # Run training for 1 epoch (using the modified Config)
    trainer.fit(epochs=1)

    # Check if checkpoint was created
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not found"

    # Run prediction
    trainer.predict()

    # Check if submission file was created
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"

    # Verify submission content format
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            print(f"Sample Submission Line: {lines[0].strip()}")
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Submission line should have at least sequence ID"

    # 6. Utility Functions Demonstration
    print("\n--- Testing Utility Functions ---")

    # Test Decoding
    # Sequence: Background(0) -> Gesture(1) -> Gesture(1) -> Background(0) -> Gesture(2) -> Gesture(2)
    # Expected: [1, 2]
    raw_preds = [0, 0, 1, 1, 1, 0, 0, 2, 2, 0]
    decoded = decode_predictions(raw_preds)
    print(f"Raw: {raw_preds} -> Decoded: {decoded}")
    assert decoded == [1, 2], f"Decoding failed. Expected [1, 2], got {decoded}"

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2]  # Distance 1 (Deletion)
    seq3 = [1, 4, 3]  # Distance 1 (Substitution)

    # Metric is sum(dist) / sum(len(truth))
    # Case 1: Pred=seq1, Truth=seq2. Dist=1. Total Truth Len=2. Score = 0.5
    score = compute_levenshtein([seq1], [seq2])
    print(f"Levenshtein Score (Pred: {seq1}, Truth: {seq2}): {score}")
    assert np.isclose(score, 0.5), f"Expected 0.5, got {score}"

    # Test Median Filter
    noisy_input = np.array([1, 1, 2, 1, 1])  # The '2' is noise
    filtered = median_filter_predictions(noisy_input, kernel_size=3)
    print(f"Noisy: {noisy_input} -> Filtered: {filtered}")
    # With kernel 3, index 2 becomes median([1, 2, 1]) = 1
    assert filtered[2] == 1, "Median filter failed to remove noise"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
