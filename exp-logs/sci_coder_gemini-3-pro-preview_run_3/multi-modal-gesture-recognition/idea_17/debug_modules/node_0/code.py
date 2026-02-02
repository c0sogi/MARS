import os
import sys
import shutil
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data, get_data_loaders, GestureDataset
from library.model import RS_KRN
from library.train import train_epoch, validate
from library.inference import predict_sequence, generate_submission


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print(">>> Step 1: Setting up configuration for demonstration...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config attributes to use the demo environment
    # We modify the class attributes directly so imported modules see the changes
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Ensure these directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce computational load for the demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.HIDDEN_DIM = 32  # Smaller model for speed
    Config.TCN_CHANNELS = 16

    # Set seed for reproducibility
    set_seed()
    print("Configuration updated for demo execution.")

    # ==========================================
    # 2. Create Mini-Datasets (Subset of Metadata)
    # ==========================================
    print("\n>>> Step 2: Creating mini-datasets for rapid testing...")

    # We will take the top N samples from the existing metadata files
    NUM_SAMPLES = 4

    # Define paths for mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test.csv")

    # Helper to create subset
    def create_subset(src, dst, n):
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take top n, or all if less than n
            df_mini = df.head(n)
            df_mini.to_csv(dst, index=False)
            print(f"Created {dst} with {len(df_mini)} samples.")
        else:
            raise FileNotFoundError(f"Source metadata {src} not found.")

    create_subset(
        os.path.join(Config.METADATA_DIR, "train.csv"), mini_train_path, NUM_SAMPLES
    )
    create_subset(
        os.path.join(Config.METADATA_DIR, "val.csv"), mini_val_path, NUM_SAMPLES
    )
    create_subset(
        os.path.join(Config.METADATA_DIR, "test.csv"), mini_test_path, NUM_SAMPLES
    )

    # Point Config to these new mini files
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # ==========================================
    # 3. Data Loading & Verification
    # ==========================================
    print("\n>>> Step 3: Verifying Data Loading...")

    # Load raw data structures (this triggers processing and caching)
    # We force load_cached_data=False initially to ensure processing logic runs,
    # but since cache dir is empty, it would process anyway.
    train_data_raw = load_data("train", load_cached_data=False)
    val_data_raw = load_data("val", load_cached_data=False)

    assert len(train_data_raw) > 0, "Train data list is empty"
    sample = train_data_raw[0]
    print(f"Loaded sample '{sample['id']}' with keys: {list(sample.keys())}")

    # Verify raw data shapes
    # Skeleton: (T, 20, 3), Audio: (T, 13), Labels: (T,)
    T = sample["skeleton"].shape[0]
    assert sample["skeleton"].shape == (T, Config.NUM_JOINTS, 3)
    assert sample["audio"].shape == (T, Config.NUM_MFCC)
    assert sample["labels"].shape == (T,)

    # Test DataLoader and Batching
    train_loader, val_loader = get_data_loaders(batch_size=Config.BATCH_SIZE)

    # Fetch one batch
    features, labels = next(iter(train_loader))

    print(f"Batch Features Shape: {features.shape}")  # (Batch, Window, InputDim)
    print(f"Batch Labels Shape: {labels.shape}")  # (Batch, Window)

    # Verify Feature Dimensions
    # InputDim = Skeleton Features (180) + Audio (13) = 193
    expected_dim = Config.SKELETON_FEATURE_DIM + Config.NUM_MFCC
    assert (
        features.shape[2] == expected_dim
    ), f"Expected feature dim {expected_dim}, got {features.shape[2]}"
    assert (
        features.shape[1] == Config.WINDOW_SIZE
    ), f"Expected window size {Config.WINDOW_SIZE}, got {features.shape[1]}"

    # ==========================================
    # 4. Model Instantiation & Forward Pass
    # ==========================================
    print("\n>>> Step 4: Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = RS_KRN().to(device)

    # Move batch to device
    features = features.to(device)

    # Forward pass
    outputs = model(features)

    # Output should be a list of tensors (one per refinement stage + encoder)
    # Config.NUM_REFINEMENT_STAGES = 2, so we expect 1 (encoder) + 2 (refinement) = 3 outputs
    expected_outputs = 1 + Config.NUM_REFINEMENT_STAGES
    assert isinstance(outputs, list), "Model output should be a list"
    assert (
        len(outputs) == expected_outputs
    ), f"Expected {expected_outputs} outputs, got {len(outputs)}"

    # Check shape of final output: (Batch, Window, NumClasses)
    final_logits = outputs[-1]
    assert final_logits.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        Config.NUM_CLASSES,
    )
    print("Model forward pass successful.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n>>> Step 5: Demonstrating Training Loop...")

    # Setup Loss and Optimizer
    from library.utils import LogSpaceSmoothingLoss

    class_weights = Config.CLASS_WEIGHTS.to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_smooth = LogSpaceSmoothingLoss(weight=Config.MSE_SMOOTHING_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_smooth, device
        )

        # Validation on full sequences
        val_score = validate(model, val_data_raw, device)

        print(
            f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Levenshtein = {val_score:.4f}"
        )

    # Save the dummy model for inference step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n>>> Step 6: Demonstrating Inference and Submission...")

    # Test single sequence prediction
    test_data_raw = load_data("test", load_cached_data=False)
    if len(test_data_raw) > 0:
        sample_test = test_data_raw[0]
        model.eval()
        prediction = predict_sequence(model, sample_test, device)
        print(f"  Single sample prediction ({sample_test['id']}): {prediction}")
        assert isinstance(prediction, list), "Prediction should be a list of IDs"

    # Generate full submission file
    generate_submission(load_cached_data=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
        print(f"  Submission file generated with {len(lines)} lines.")
        print(f"  First line: {lines[0].strip()}")
        assert len(lines) == len(test_data_raw), "Submission line count mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n>>> Demo execution completed successfully.")


if __name__ == "__main__":
    main()
