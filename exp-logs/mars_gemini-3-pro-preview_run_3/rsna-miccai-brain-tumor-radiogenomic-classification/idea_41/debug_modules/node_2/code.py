import os
import sys
import pandas as pd
import numpy as np
import torch
import importlib
import shutil

# Import the provided library modules
import library.config as config
import library.utils as utils

# We will import the other modules dynamically after patching the configuration
# to ensure they pick up the modified paths.


def create_subset_metadata(n_samples=10):
    """
    Creates small subsets of the original metadata files to speed up
    demonstration and testing.
    """
    print(f"Creating metadata subsets with {n_samples} samples...")

    # Define paths for subsets
    subset_train_path = os.path.join(config.WORKING_DIR, "subset_train.parquet")
    subset_val_path = os.path.join(config.WORKING_DIR, "subset_val.parquet")
    subset_test_path = os.path.join(config.WORKING_DIR, "subset_test.parquet")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Load original metadata
    train_df = pd.read_parquet(config.TRAIN_META_PATH)
    val_df = pd.read_parquet(config.VAL_META_PATH)
    test_df = pd.read_parquet(config.TEST_META_PATH)

    # Create subsets
    train_sub = train_df.head(n_samples).copy()
    val_sub = val_df.head(n_samples).copy()
    test_sub = test_df.head(n_samples).copy()

    # Save subsets
    train_sub.to_parquet(subset_train_path, index=False)
    val_sub.to_parquet(subset_val_path, index=False)
    test_sub.to_parquet(subset_test_path, index=False)

    print(f"Subsets saved to {config.WORKING_DIR}")
    return subset_train_path, subset_val_path, subset_test_path


def patch_and_reload_libraries(train_path, val_path, test_path):
    """
    Updates the configuration module with new paths and reloads dependent
    modules to apply changes.
    """
    print("Patching configuration with subset paths...")

    # Update config module variables
    config.TRAIN_META_PATH = train_path
    config.VAL_META_PATH = val_path
    config.TEST_META_PATH = test_path

    # Force reload of modules that import these constants
    # We need to import them first to reload them
    global data_loader, model_lib, train_lib, predict_lib

    import library.data_loader as data_loader
    import library.model as model_lib
    import library.train as train_lib
    import library.predict as predict_lib

    print("Reloading modules...")
    importlib.reload(data_loader)
    # Reload model_lib first so dependents (train_lib, predict_lib) pick up the new class definition
    importlib.reload(model_lib)
    importlib.reload(train_lib)
    importlib.reload(predict_lib)


def verify_data_loading():
    """
    Verifies that the DataLoader produces batches of the correct shape.
    """
    print("\n=== Verifying Data Loading ===")

    # Use a small batch size for verification
    batch_size = 4

    # Get dataloaders (force reload to process the subset metadata)
    train_loader, val_loader, test_loader, test_ids = data_loader.get_dataloaders(
        batch_size=batch_size,
        load_cached_data=False,  # Force processing of our new subsets
    )

    print(f"Train loader length: {len(train_loader)}")

    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    x_even, x_odd = inputs

    # Expected shapes
    # x_even: (B, 64, 224, 224)
    # targets: (B,)
    print(f"x_even shape: {x_even.shape}")
    print(f"x_odd shape: {x_odd.shape}")
    print(f"targets shape: {targets.shape}")

    assert x_even.shape == (
        batch_size,
        64,
        224,
        224,
    ), f"Unexpected shape for x_even: {x_even.shape}"
    assert x_odd.shape == (
        batch_size,
        64,
        224,
        224,
    ), f"Unexpected shape for x_odd: {x_odd.shape}"
    assert targets.shape == (
        batch_size,
    ), f"Unexpected shape for targets: {targets.shape}"

    print("Data loading verification passed.")
    return train_loader


def verify_model_architecture():
    """
    Verifies the model forward pass and output shape.
    """
    print("\n=== Verifying Model Architecture ===")

    device = utils.get_device()
    model = model_lib.SiameseEfficientNet().to(device)

    # Create dummy input
    batch_size = 2
    dummy_even = torch.randn(batch_size, 64, 224, 224).to(device)
    dummy_odd = torch.randn(batch_size, 64, 224, 224).to(device)

    # Forward pass
    logits = model(dummy_even, dummy_odd)

    print(f"Logits shape: {logits.shape}")

    # Expected output: (B, 1)
    assert logits.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {logits.shape}"

    print("Model architecture verification passed.")


def run_full_pipeline_demo():
    """
    Runs the complete training and inference pipeline using the provided library functions.
    """
    print("\n=== Running Full Pipeline Demo ===")

    # 1. Run Training
    # This uses run_training from library.train
    # It will use the subset data we configured, so 15 epochs will be very fast.
    print("Starting training loop...")
    train_lib.run_training(load_cached_data=True)
    # Note: We set load_cached_data=True here because verify_data_loading()
    # already generated and cached the subset data in the ./working/idea_41 directory.

    # Verify model was saved
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file was not saved!"
    print(f"Model saved successfully at {config.MODEL_SAVE_PATH}")

    # 2. Run Inference
    # This uses generate_submission from library.predict
    print("Starting inference...")
    predict_lib.generate_submission(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated!"

    # Check submission content
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission file content sample:")
    print(sub_df.head())

    # Verify rows match subset size (10)
    assert len(sub_df) == 10, f"Expected 10 predictions, got {len(sub_df)}"
    assert "BraTS21ID" in sub_df.columns
    assert "MGMT_value" in sub_df.columns

    print("Pipeline demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    config.seed_everything(42)

    # 1. Prepare Data Subsets (Optimization for Speed)
    t_path, v_path, te_path = create_subset_metadata(n_samples=10)

    # 2. Patch Config and Reload Libraries
    patch_and_reload_libraries(t_path, v_path, te_path)

    # 3. Verify Components
    verify_data_loading()
    verify_model_architecture()

    # 4. Run Pipeline
    run_full_pipeline_demo()

    print("\nAll demonstrations passed!")
