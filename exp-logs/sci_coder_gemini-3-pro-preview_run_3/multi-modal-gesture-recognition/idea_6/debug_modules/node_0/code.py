import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Import from the provided library
from library import config, utils, model, loss, data_loader, train


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Environment
    # --------------------
    utils.set_seed(42)
    device = config.get_device()
    print(f"Device: {device}")

    # Define a temporary directory for this demo to avoid overwriting real work
    demo_base_dir = "./working/demo_execution"
    if os.path.exists(demo_base_dir):
        shutil.rmtree(demo_base_dir)
    os.makedirs(demo_base_dir, exist_ok=True)

    # 2. Configuration Overrides (Optimize for Speed)
    # -----------------------------------------------
    print("\n[Step 1] Configuring for fast demonstration...")

    # Paths
    config.WORKING_DIR = demo_base_dir
    config.CACHE_DIR = os.path.join(demo_base_dir, "cache")
    config.MODEL_SAVE_PATH = os.path.join(demo_base_dir, "outputs", "demo_model.pth")
    config.SUBMISSION_FILE = os.path.join(demo_base_dir, "submission.csv")

    # Ensure directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # Training Hyperparameters for Demo
    config.BATCH_SIZE = 2
    config.NUM_EPOCHS = 1
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple script debugging
    config.WINDOW_SIZE = 32  # Smaller window for speed
    config.STRIDE = 32

    # 3. Data Preparation (Mini Subset)
    # ---------------------------------
    print("\n[Step 2] Creating mini-datasets from metadata...")

    # Load original metadata
    try:
        df_train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
        df_val_full = pd.read_csv(config.VAL_METADATA_PATH)
        df_test_full = pd.read_csv(config.TEST_METADATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: Metadata files not found. Ensure ./metadata exists. {e}")
        return

    # Take a tiny subset (e.g., 2 samples each)
    mini_train = df_train_full.head(2)
    mini_val = df_val_full.head(2)
    mini_test = df_test_full.head(2)

    # Save mini metadata
    mini_train_path = os.path.join(demo_base_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_base_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_base_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update config to point to mini metadata
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    print(f"  Created mini metadata at {demo_base_dir}")

    # 4. Component Validation: Data Loader
    # ------------------------------------
    print("\n[Step 3] Validating Data Loader...")

    # Initialize loaders (load_cached_data=False forces processing of the mini subset)
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        load_cached_data=False
    )

    # Fetch one batch from training loader
    try:
        features_batch, targets_batch = next(iter(train_loader))
        print(f"  Train Batch Features Shape: {features_batch.shape}")
        print(f"  Train Batch Targets Shape: {targets_batch.shape}")

        # Assertions
        # Features: (Batch, Time, InputDim)
        assert features_batch.dim() == 3, "Features should be 3D tensor"
        assert (
            features_batch.shape[2] == config.INPUT_DIM
        ), f"Input dim mismatch. Expected {config.INPUT_DIM}, got {features_batch.shape[2]}"
        # Targets: (Batch, Time)
        assert targets_batch.dim() == 2, "Targets should be 2D tensor"
        assert (
            targets_batch.shape[0] == features_batch.shape[0]
        ), "Batch size mismatch between features and targets"

    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 5. Component Validation: Model Architecture
    # -------------------------------------------
    print("\n[Step 4] Validating Model Architecture...")

    net = model.IterativeCascadedNet().to(device)

    # Move batch to device
    features_batch = features_batch.to(device)
    targets_batch = targets_batch.to(device)

    # Forward pass
    outputs = net(features_batch)

    # Assertions
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, "Model should return outputs for 3 stages"

    stage1_logits = outputs[0]
    # Shape: (Batch, Classes, Time)
    assert stage1_logits.shape == (
        features_batch.shape[0],
        config.NUM_CLASSES,
        features_batch.shape[1],
    ), f"Output shape mismatch. Expected {(features_batch.shape[0], config.NUM_CLASSES, features_batch.shape[1])}, got {stage1_logits.shape}"

    print("  Forward pass successful. Output shapes verified.")

    # 6. Component Validation: Loss Function
    # --------------------------------------
    print("\n[Step 5] Validating Loss Function...")

    criterion = loss.CascadedLoss()
    total_loss = criterion(outputs, targets_batch)

    print(f"  Calculated Loss: {total_loss.item():.4f}")

    assert torch.is_tensor(total_loss), "Loss should be a tensor"
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert total_loss.item() > 0, "Loss should be positive"

    # 7. Component Validation: Utilities
    # ----------------------------------
    print("\n[Step 6] Validating Utility Functions...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]
    dist = utils.compute_levenshtein(seq1, seq2)
    assert dist == 1, f"Levenshtein distance incorrect. Expected 1, got {dist}"

    # Test Decode Predictions
    # Create dummy probabilities (Time, Classes)
    dummy_probs = torch.zeros((10, config.NUM_CLASSES))
    # Set class 1 for first 5 frames, class 2 for next 5
    dummy_probs[0:5, 1] = 1.0
    dummy_probs[5:10, 2] = 1.0

    decoded = utils.decode_predictions(dummy_probs, threshold=2)
    # Expect [1, 2]
    assert decoded == [1, 2], f"Decoding failed. Expected [1, 2], got {decoded}"

    print("  Utility functions verified.")

    # 8. Integration Test: Full Training Loop
    # ---------------------------------------
    print("\n[Step 7] Running Full Training Loop (1 Epoch)...")

    # We call the provided train.run_training function.
    # It will re-initialize loaders (using the cached mini-data this time) and run the loop.
    train.run_training(num_epochs=1, load_cached_data=True)

    # Verify outputs
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"  Success: Model saved to {config.MODEL_SAVE_PATH}")
    else:
        # Note: Model is only saved if validation score improves.
        # With random init and 1 epoch, it might not "improve" over inf if logic is strict,
        # but the code sets best_score = inf initially, so any valid score saves it.
        # However, if validation set is empty or score is NaN, it might fail.
        # Let's check if submission file exists as a proxy for completion.
        pass

    if os.path.exists(config.SUBMISSION_FILE):
        print(f"  Success: Submission file generated at {config.SUBMISSION_FILE}")

        # Validate submission format
        sub_df = pd.read_csv(config.SUBMISSION_FILE, header=None)
        # Should have 2 columns (if labels exist) or just rows.
        # The format is SessionID,Label1,Label2...
        # Let's just check row count matches mini_test
        assert len(sub_df) == len(
            mini_test
        ), f"Submission row count mismatch. Expected {len(mini_test)}, got {len(sub_df)}"
        print("  Submission format check passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
