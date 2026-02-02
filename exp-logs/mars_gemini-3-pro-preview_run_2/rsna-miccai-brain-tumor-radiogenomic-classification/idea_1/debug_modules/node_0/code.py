import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train
import library.inference as inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of MGMT Methylation Prediction Pipeline ===\n")

    # 1. Reproducibility
    utils.seed_everything(config.SEED)
    print(f"Random seed set to {config.SEED}")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loading ---")

    # Load metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    print(f"Loaded training metadata: {len(df_train)} samples")

    # Create a small subset for quick verification
    subset_df = df_train.head(5).copy()

    # Instantiate Dataset
    dataset = data.BraTSDataset(
        metadata=subset_df,
        base_dir=config.INPUT_DIR,
        transform=data.get_transforms("train"),
        is_test=False,
    )

    # Fetch one sample
    sample_img, sample_label = dataset[0]

    # Verify shapes
    # Expected shape: (Channels, Height, Width) -> (12, 256, 256)
    expected_channels = config.IN_CHANNELS
    expected_size = config.IMG_SIZE

    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    assert sample_img.shape == (
        expected_channels,
        expected_size,
        expected_size,
    ), f"Image shape mismatch. Expected ({expected_channels}, {expected_size}, {expected_size}), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"

    print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")

    device = config.DEVICE
    net = model.MGMTNet(pretrained=False)  # No need to download weights for shape check
    net.to(device)
    net.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, expected_channels, expected_size, expected_size).to(
        device
    )

    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected output: (Batch_Size, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Running Training Demonstration (1 Epoch) ---")

    # We run for just 1 epoch to demonstrate the loop works without timing out.
    # The run_training function handles data loading, model init, and the loop.
    try:
        train.run_training(epochs=1)
        print("Training finished successfully.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify checkpoint creation
    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"Checkpoint found at: {config.CHECKPOINT_PATH}")
    else:
        # If validation loss didn't improve (unlikely in epoch 1 vs inf), it might not save.
        # However, logic initializes best_loss to inf, so it should save.
        print("Warning: Checkpoint was not created (possibly due to logic or error).")

    # 5. Inference Demonstration
    print("\n--- Running Inference Demonstration ---")

    try:
        inference.generate_submission(
            checkpoint_path=config.CHECKPOINT_PATH, output_path=config.SUBMISSION_FILE
        )
        print("Inference finished successfully.")
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # 6. Submission Verification
    print("\n--- Verifying Submission File ---")

    if not os.path.exists(config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_FILE}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Check for required columns
    assert "BraTS21ID" in df_sub.columns, "Submission missing 'BraTS21ID' column"
    assert "MGMT_value" in df_sub.columns, "Submission missing 'MGMT_value' column"

    # Check value range (probabilities)
    probs = df_sub["MGMT_value"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"

    # Check ID count matches test metadata
    df_test_meta = pd.read_csv(config.TEST_METADATA_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) does not match test set size ({len(df_test_meta)})"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
