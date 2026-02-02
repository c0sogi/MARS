import os
import sys
import shutil
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library import config, utils, data_loader, model, train


def main():
    print("=== Starting Library Usage Demonstration ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override configuration for a quick demonstration run
    print("Configuring environment for demo...")

    # Create a separate working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update config paths
    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    config.LOG_DIR = os.path.join(DEMO_DIR, "logs")

    # Create subdirectories
    for d in [
        config.CACHE_DIR,
        config.CHECKPOINT_DIR,
        config.SUBMISSION_DIR,
        config.LOG_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # Set hyperparameters for speed
    config.MAX_SAMPLES = 20  # Use only 20 samples for training/val/test
    config.BATCH_SIZE = 4  # Small batch size
    config.NUM_EPOCHS = 2  # Only 2 epochs
    config.DEBUG = True

    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loader Demonstration
    # ==========================================
    print("\n[Step 1] Initializing Data Loaders...")

    # We use get_loaders which handles dataset instantiation and caching
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.BATCH_SIZE, max_samples=config.MAX_SAMPLES
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Validate a single training batch
    print("Validating training batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    assert "skeleton" in batch, "Batch missing 'skeleton' key"
    assert "audio" in batch, "Batch missing 'audio' key"
    assert "labels" in batch, "Batch missing 'labels' key"

    # Check shapes (Batch, Time, Feat)
    skel_shape = batch["skeleton"].shape
    audio_shape = batch["audio"].shape
    labels_shape = batch["labels"].shape

    print(f"Skeleton Shape: {skel_shape}")  # Expected: (B, WindowSize, SkelDim)
    print(f"Audio Shape: {audio_shape}")  # Expected: (B, WindowSize, AudioDim)
    print(f"Labels Shape: {labels_shape}")  # Expected: (B, WindowSize)

    assert skel_shape[0] == config.BATCH_SIZE, f"Batch size mismatch: {skel_shape[0]}"
    assert skel_shape[1] == config.WINDOW_SIZE, f"Window size mismatch: {skel_shape[1]}"
    assert labels_shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
    ), "Labels shape mismatch"

    # ==========================================
    # 3. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[Step 2] Instantiating SKD-GN Model...")

    net = model.SKD_GN().to(device)

    # Move batch to device for testing
    skel_input = batch["skeleton"].to(device)
    audio_input = batch["audio"].to(device)

    print("Running forward pass...")
    outputs = net(skel_input, audio_input)

    # Validate outputs
    assert (
        "p1" in outputs and "p2" in outputs and "p3" in outputs
    ), "Model output missing stages"

    # Check output shape: (Batch, Time, NumClasses)
    p3_shape = outputs["p3"].shape
    print(f"Output P3 Shape: {p3_shape}")

    assert p3_shape[0] == config.BATCH_SIZE
    assert p3_shape[1] == config.WINDOW_SIZE
    assert p3_shape[2] == config.NUM_CLASSES

    # ==========================================
    # 4. Loss Function Usage
    # ==========================================
    print("\n[Step 3] Calculating Loss...")

    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = train.CombinedLoss(
        weight=class_weights,
        smoothing_lambda=config.SMOOTHING_LAMBDA,
        smoothing_threshold=config.SMOOTHING_THRESHOLD,
    )

    labels_input = batch["labels"].to(device)

    # Calculate loss on the dummy batch
    loss_val = criterion(outputs["p3"], labels_input)
    print(f"Calculated Loss: {loss_val.item():.4f}")

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() > 0, "Loss should be positive"

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[Step 4] Running Training Loop...")

    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_score = float("inf")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train.train_epoch(net, train_loader, optimizer, criterion, device)

        # Validate
        # Note: Validation metric is Levenshtein distance (lower is better)
        val_score = train.validate(net, val_loader, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(net.state_dict(), best_model_path)
            print("  -> Saved new best model.")

    assert os.path.exists(best_model_path), "Best model checkpoint was not created."

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[Step 5] Generating Submission...")

    # Load best model
    net.load_state_dict(torch.load(best_model_path))

    submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    train.generate_submission(net, test_loader, device, submission_file)

    # Verify submission file
    assert os.path.exists(submission_file), "Submission file not found."

    # Check content format
    with open(submission_file, "r") as f:
        lines = f.readlines()
        print(f"Generated {len(lines)} prediction lines.")
        if len(lines) > 0:
            print(f"Sample prediction: {lines[0].strip()}")
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Invalid CSV format"
            # First part should be sample ID (e.g., Sample00001 or similar)
            # Remaining parts are gesture IDs

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
