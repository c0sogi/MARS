import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config, utils, data_loader, model, losses, trainer


def main():
    print("=== Starting Demo Script for Gesture Recognition Pipeline ===")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override config parameters to run a quick test
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    config.NUM_EPOCHS = 2  # Train for only 2 epochs
    config.BATCH_SIZE = 4
    config.WORKING_DIR = "./working/demo_execution"
    config.BEST_MODEL_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.SUBMISSION_PATH = os.path.join(
        config.WORKING_DIR, "submission", "submission.csv"
    )

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Set seed for reproducibility
    utils.set_seed(42)
    print("Configuration updated for demo mode.")

    # ==========================================
    # 2. Testing Utilities
    # ==========================================
    print("\n[2] Testing Utility Functions...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_equal = utils.levenshtein_distance(seq1, seq2)
    assert (
        dist_equal == 0
    ), f"Levenshtein distance for equal seqs should be 0, got {dist_equal}"

    seq3 = [1, 2]
    dist_diff = utils.levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Levenshtein distance should be 1, got {dist_diff}"
    print("Levenshtein distance check passed.")

    # Test Run-Length Encoding
    preds = [1, 1, 1, 2, 2, 0, 0, 3]
    segments = utils.run_length_encoding(preds)
    # Expected: [(1, 0, 2), (2, 3, 4), (0, 5, 6), (3, 7, 7)]
    assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"
    assert segments[0] == (1, 0, 2), f"First segment mismatch: {segments[0]}"
    print("Run-Length Encoding check passed.")

    # ==========================================
    # 3. Testing Data Loading
    # ==========================================
    print("\n[3] Testing Data Loader...")

    # Force re-creation of cache for demo
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.BATCH_SIZE, debug=config.DEBUG
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    skeleton = batch["skeleton"]
    audio = batch["audio"]
    labels = batch["label"]

    print(
        f"Batch Shapes -> Skeleton: {skeleton.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Validation of shapes
    # Skeleton: (Batch, Window, 180)
    assert skeleton.dim() == 3 and skeleton.shape[2] == 180, "Incorrect skeleton shape"
    # Audio: (Batch, Window, 13)
    assert audio.dim() == 3 and audio.shape[2] == config.N_MFCC, "Incorrect audio shape"
    # Labels: (Batch, Window)
    assert labels.dim() == 2, "Incorrect label shape"

    print("Data Loader check passed.")

    # ==========================================
    # 4. Testing Model Architecture
    # ==========================================
    print("\n[4] Testing Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.ANG_KN().to(device)

    # Move batch to device
    skel_in = skeleton.to(device)
    audio_in = audio.to(device)

    # Forward pass
    outputs = net(skel_in, audio_in)

    # Check outputs
    assert (
        "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    ), "Model output missing stages"

    logits_s3 = outputs["stage3"]
    # Shape should be (Batch, Window, NumClasses)
    assert logits_s3.shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
        config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(config.BATCH_SIZE, config.WINDOW_SIZE, config.NUM_CLASSES)}, got {logits_s3.shape}"

    print("Model forward pass check passed.")

    # ==========================================
    # 5. Testing Loss Function
    # ==========================================
    print("\n[5] Testing Loss Function...")

    criterion = losses.CascadedLoss().to(device)
    targets = labels.to(device)

    loss_val, loss_dict = criterion(outputs, targets)

    print(f"Calculated Loss: {loss_val.item():.4f}")
    print(f"Loss Components: {loss_dict}")

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val > 0, "Loss should be positive"

    print("Loss function check passed.")

    # ==========================================
    # 6. Testing Trainer (Training Loop)
    # ==========================================
    print("\n[6] Testing Trainer (Fit & Predict)...")

    # Initialize Trainer
    pipeline = trainer.Trainer(device=device)

    # Run Training (Fit)
    # Using the loaders and dataset obtained earlier
    pipeline.fit(train_loader, val_loader, val_loader.dataset, epochs=config.NUM_EPOCHS)

    # Check if model was saved
    assert os.path.exists(config.BEST_MODEL_PATH), "Best model file was not saved."
    print(f"Model saved successfully at {config.BEST_MODEL_PATH}")

    # Run Prediction
    print("Running prediction on test subset...")
    predictions = pipeline.predict(test_loader.dataset)

    # Verify predictions
    assert len(predictions) > 0, "No predictions generated."
    sample_id = list(predictions.keys())[0]
    pred_seq = predictions[sample_id]

    print(f"Sample Prediction for {sample_id}: {pred_seq}")
    assert isinstance(pred_seq, list), "Prediction should be a list of IDs"

    # ==========================================
    # 7. Generating Submission
    # ==========================================
    print("\n[7] Generating Submission File...")

    lines = []
    for sid, p_ids in predictions.items():
        pred_str = ",".join(map(str, p_ids))
        lines.append(f"{sid},{pred_str}")

    with open(config.SUBMISSION_PATH, "w") as f:
        f.write("\n".join(lines))

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."
    print(f"Submission file created at {config.SUBMISSION_PATH}")

    # Print first few lines of submission
    with open(config.SUBMISSION_PATH, "r") as f:
        print("--- Submission Head ---")
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            print(line.strip())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
