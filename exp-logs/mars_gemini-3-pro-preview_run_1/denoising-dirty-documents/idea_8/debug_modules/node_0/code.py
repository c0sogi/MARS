import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib


def main():
    print("Initializing Demonstration...")

    # 1. Setup & Configuration Overrides for Speed
    # We monkey-patch the EPOCHS in the train module to run only 1 epoch for demonstration.
    # Note: We must patch it in the module where it is used (library.train),
    # because it was imported via 'from ... import EPOCHS'.
    train_lib.EPOCHS = 1

    # Set a fixed seed
    utils.seed_everything(42)

    print("\n=== 1. Verifying Dataset and DataLoader ===")
    # Initialize dataset for training
    # We use load_cached_data=False to force reading from metadata/images initially to verify that path
    train_ds = dataset.TextDenoisingDataset(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    print(f"Train Dataset Size: {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset should not be empty."

    # Fetch one sample
    noisy_t, clean_t = train_ds[0]
    print(f"Sample Noisy Shape: {noisy_t.shape}")
    print(f"Sample Clean Shape: {clean_t.shape}")

    # Assertions for Train Mode (Patching)
    # Shape should be (1, PATCH_SIZE, PATCH_SIZE)
    expected_shape = (1, config.PATCH_SIZE, config.PATCH_SIZE)
    assert (
        noisy_t.shape == expected_shape
    ), f"Expected noisy shape {expected_shape}, got {noisy_t.shape}"
    assert (
        clean_t.shape == expected_shape
    ), f"Expected clean shape {expected_shape}, got {clean_t.shape}"

    # Verify DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=True
    )

    # Fetch a batch from train loader
    batch_noisy, batch_clean = next(iter(train_loader))
    print(f"Batch Noisy Shape: {batch_noisy.shape}")

    # Assertions for Batch
    # Shape: (BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    assert batch_noisy.shape[0] == config.BATCH_SIZE, "Incorrect batch size."
    assert batch_noisy.shape[1] == 1, "Incorrect channel dimension."

    print("Dataset and DataLoader verification passed.")

    print("\n=== 2. Verifying Model Architecture ===")
    # Instantiate Model
    net = model_lib.ASPPShallowUNet().to(config.DEVICE)

    # Create dummy input: (Batch=2, Channels=1, H=160, W=160)
    dummy_input = torch.randn(2, 1, 160, 160).to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert (
        output.shape == dummy_input.shape
    ), "Model input and output shapes must match."
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output must be in [0, 1] (Sigmoid)."

    print("Model architecture verification passed.")

    print("\n=== 3. Running Training Loop (1 Epoch) ===")
    # Define a specific seed for this run
    demo_seed = 42

    # Run training
    # This will run for 1 epoch (due to patch) and save 'model_seed_42.pth'
    best_rmse = train_lib.train_model(demo_seed, load_cached_data=True, patience=1)

    print(f"Training completed. Best RMSE: {best_rmse}")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(config.WORKING_DIR, f"model_seed_{demo_seed}.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"

    # Verify Checkpoint Content
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    assert "state_dict" in ckpt, "Checkpoint missing state_dict."
    assert "best_rmse" in ckpt, "Checkpoint missing best_rmse."

    print("Training loop verification passed.")

    print("\n=== 4. Running Inference ===")
    # Generate predictions using the model trained in step 3
    # We pass the list of seeds we trained (just one here)
    inference_lib.generate_predictions(seeds=[demo_seed], load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Validate Submission Format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert list(df_sub.columns) == ["id", "value"], "Incorrect submission columns."
    assert not df_sub.isnull().values.any(), "Submission contains null values."

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    assert len(str(sample_id).split("_")) == 3, f"Incorrect ID format: {sample_id}"

    # Check Value range
    min_val = df_sub["value"].min()
    max_val = df_sub["value"].max()
    print(f"Prediction Range: [{min_val}, {max_val}]")
    assert min_val >= 0 and max_val <= 1, "Predictions out of range [0, 1]."

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
