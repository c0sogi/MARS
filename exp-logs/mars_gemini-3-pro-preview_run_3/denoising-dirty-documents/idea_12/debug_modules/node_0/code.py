import os
import torch
import numpy as np
import pandas as pd
import glob
import shutil

# Import from the provided library
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model
import library.train as train
import library.inference as inference


def run_demo():
    print("=== Starting Denoising Pipeline Demo ===")

    # 1. Setup and Reproducibility
    print("\n[1] Setting up environment...")
    utils.seed_everything(config.SEED)

    # Override configuration for speed in this demo
    # We patch the module-level variables since they are imported directly in the library files
    print("    Patching configuration for fast execution...")
    train.NUM_EPOCHS_STAGE_1 = 1
    train.NUM_EPOCHS_STAGE_2 = 1
    train.BATCH_SIZE = 4

    # Increase stride to reduce number of patches extracted
    dataset.STRIDE_SPARSE = 200
    dataset.STRIDE_DENSE = 200

    # Define working paths
    demo_working_dir = config.WORKING_DIR
    demo_submission_path = os.path.join(demo_working_dir, "demo_submission.csv")

    # 2. Demonstrate Dataset Loading and Processing
    print("\n[2] Demonstrating Dataset components...")

    # Load a small subset of images directly
    limit = 5
    print(f"    Loading {limit} images for inspection...")
    inputs, targets = dataset.load_images(config.TRAIN_METADATA_PATH, limit=limit)

    assert len(inputs) == limit, f"Expected {limit} input images, got {len(inputs)}"
    assert len(targets) == limit, f"Expected {limit} target images, got {len(targets)}"
    assert inputs[0].dtype == np.float32, "Images should be float32"
    assert inputs[0].max() <= 1.0, "Images should be normalized to [0, 1]"

    # Extract patches manually to verify shape
    print("    Extracting patches...")
    patches = dataset.extract_patches(inputs, config.PATCH_SIZE, stride=200)
    print(f"    Extracted patches shape: {patches.shape}")

    # Verify patch shape: (N, 1, H, W)
    assert len(patches.shape) == 4
    assert patches.shape[1] == 1
    assert patches.shape[2] == config.PATCH_SIZE
    assert patches.shape[3] == config.PATCH_SIZE

    # Instantiate Dataset class
    ds = dataset.DenoisingDataset(
        patches, patches, augment=True
    )  # Using inputs as targets for dummy check
    sample_x, sample_y = ds[0]

    assert isinstance(sample_x, torch.Tensor)
    assert sample_x.shape == (1, config.PATCH_SIZE, config.PATCH_SIZE)
    print("    Dataset verification successful.")

    # 3. Demonstrate Model Architecture
    print("\n[3] Demonstrating Model architecture...")
    net = model.CAResDnCNN().to(config.DEVICE)

    # Create a dummy batch
    dummy_input = torch.randn(2, 1, config.PATCH_SIZE, config.PATCH_SIZE).to(
        config.DEVICE
    )

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"
    print("    Model forward pass successful.")

    # 4. Demonstrate Training Pipeline
    print("\n[4] Running Training Pipeline (Fast Mode)...")

    # clean up previous demo artifacts if any
    final_model_path = os.path.join(demo_working_dir, "final_model.pth")
    if os.path.exists(final_model_path):
        os.remove(final_model_path)

    # Run training with limit to ensure speed
    trained_model = train.train_model(limit=5)

    assert os.path.exists(final_model_path), "Final model file was not saved."
    print(f"    Training complete. Model saved to {final_model_path}")

    # 5. Demonstrate Inference and Submission Generation
    print("\n[5] Running Inference and Submission Generation...")

    # Run generation
    inference.generate_submission(
        model_path=final_model_path, output_path=demo_submission_path
    )

    assert os.path.exists(demo_submission_path), "Submission file not created."

    # Validate submission format
    print("    Validating submission format...")
    df_sub = pd.read_csv(demo_submission_path)

    # Check header
    assert list(df_sub.columns) == [
        "id",
        "value",
    ], f"Incorrect columns: {df_sub.columns}"

    # Check content
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        df_sub["value"].dtype == float or df_sub["value"].dtype == np.float64
    ), "Value column should be float"

    # Check ID format (e.g., 110_1_1)
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) == 3, f"ID format incorrect: {sample_id}"

    print(f"    Submission generated with {len(df_sub)} rows.")
    print(f"    First 3 rows:\n{df_sub.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
