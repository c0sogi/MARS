import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import library modules
from library import config, utils, data, model, train, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1] Setting up configuration and temporary directories...")

    # Define a temporary working directory for this demo
    DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override global config variables to ensure speed and isolation
    config.IDEA_NAME = "demo_execution"
    config.CACHE_DIR = DEMO_DIR
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Use 0 for simple debugging/demo
    config.EPOCHS = 1
    config.EARLY_STOPPING_PATIENCE = 1

    # Redirect output paths
    config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Set seed for reproducibility
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Demo Directory: {DEMO_DIR}")

    # -------------------------------------------------------------------------
    # 2. Create Subset Metadata (for speed)
    # -------------------------------------------------------------------------
    print("\n[2] Creating metadata subsets for rapid testing...")

    # Load original metadata
    full_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val = pd.read_csv(config.VAL_METADATA_PATH)
    full_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subsets (e.g., 8 train, 4 val, 4 test)
    demo_train_df = full_train.head(8).copy()
    demo_val_df = full_val.head(4).copy()
    demo_test_df = full_test.head(4).copy()

    # Save to demo directory
    demo_train_path = os.path.join(DEMO_DIR, "demo_train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "demo_val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "demo_test.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    print(f"    Created demo_train.csv ({len(demo_train_df)} rows)")
    print(f"    Created demo_val.csv ({len(demo_val_df)} rows)")
    print(f"    Created demo_test.csv ({len(demo_test_df)} rows)")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading logic...")

    # Initialize DataLoader with the subset
    # This will trigger anchor calculation for the 8 subjects (fast)
    train_loader = data.get_dataloader(
        metadata_path=demo_train_path,
        batch_size=config.BATCH_SIZE,
        is_train=True,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        load_cached_data=False,  # Force recompute for demo isolation
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"    Input Batch Shape: {inputs.shape}")
    print(f"    Target Batch Shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, Channels, Height, Width) -> (4, 20, 224, 224)
    expected_channels = config.INPUT_CHANNELS  # 20
    assert inputs.shape == (
        config.BATCH_SIZE,
        expected_channels,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Shape mismatch! Expected {(config.BATCH_SIZE, expected_channels, config.IMG_SIZE, config.IMG_SIZE)}, got {inputs.shape}"

    assert targets.shape[0] == config.BATCH_SIZE, "Target batch size mismatch"
    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    net = model.AsymmetricEfficientNet().to(device)

    # Pass the batch from step 3
    inputs = inputs.to(device)
    outputs = net(inputs)

    print(f"    Model Output Shape: {outputs.shape}")

    # Assertions
    # Expected: (Batch, Num_Classes) -> (4, 1)
    assert outputs.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Output shape mismatch! Expected {(config.BATCH_SIZE, config.NUM_CLASSES)}, got {outputs.shape}"

    assert outputs.requires_grad, "Model output should track gradients for training."
    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Pipeline (1 Epoch)...")

    # Run training using the subset metadata
    # Note: We pass epochs explicitly because the default arg in run_training
    # might have been bound to the original config.EPOCHS at import time.
    train.run_training(
        train_metadata_path=demo_train_path,
        val_metadata_path=demo_val_path,
        epochs=1,
        patience=1,
    )

    # Verify model file was created
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"    Success: Model saved at {config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Training completed but model file was not found!")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    predict.predict_submission(
        test_metadata_path=demo_test_path,
        model_path=config.MODEL_SAVE_PATH,
        submission_output_path=config.SUBMISSION_FILE,
        batch_size=config.BATCH_SIZE,
        device=device,
    )

    # Verify submission file
    if os.path.exists(config.SUBMISSION_FILE):
        sub_df = pd.read_csv(config.SUBMISSION_FILE)
        print(f"    Submission generated with {len(sub_df)} rows.")
        print(f"    Head:\n{sub_df.head()}")

        # Assertions
        assert len(sub_df) == len(
            demo_test_df
        ), f"Submission length mismatch! Expected {len(demo_test_df)}, got {len(sub_df)}"
        assert (
            "BraTS21ID" in sub_df.columns and "MGMT_value" in sub_df.columns
        ), "Submission columns mismatch"
    else:
        raise FileNotFoundError("Inference completed but submission file not found!")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
