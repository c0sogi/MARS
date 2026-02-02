import os
import numpy as np
import pandas as pd
import torch
import shutil

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import train
from library import predict


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # -------------------------------------------------------------------------
    print("1. Configuring environment for demo execution...")

    # Define a demo-specific working directory
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config parameters to ensure speed
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_working_dir

    # Update paths derived from WORKING_DIR
    config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "demo_model.pth")
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")
    config.TRAIN_PATCHES_CACHE = os.path.join(demo_working_dir, "train_patches.npy")
    config.TRAIN_TARGETS_CACHE = os.path.join(demo_working_dir, "train_targets.npy")
    config.VAL_PATCHES_CACHE = os.path.join(demo_working_dir, "val_patches.npy")
    config.VAL_TARGETS_CACHE = os.path.join(demo_working_dir, "val_targets.npy")

    # Reduce Model Complexity
    config.RDN_NUM_BLOCKS = 1  # Only 1 block
    config.RDN_LAYERS_PER_BLOCK = 2  # Only 2 layers per block
    config.RDN_NUM_FEATURES = 16  # Fewer features
    config.RDN_GROWTH_RATE = 16  # Smaller growth rate

    # Reduce Data Size
    config.PATCH_SIZE = 32  # Smaller patches
    config.STRIDE = 200  # Large stride = fewer patches = faster

    # Reduce Training Duration
    config.NUM_EPOCHS = 1  # Single epoch
    config.BATCH_SIZE = 8

    # Create a temporary test metadata file with fewer samples for faster prediction
    # We'll take the first 3 rows from the original test metadata
    original_test_df = pd.read_csv(config.TEST_METADATA_PATH)
    demo_test_df = original_test_df.head(3)
    demo_test_metadata_path = os.path.join(demo_working_dir, "temp_test.csv")
    demo_test_df.to_csv(demo_test_metadata_path, index=False)

    # Override test metadata path in config for prediction step
    config.TEST_METADATA_PATH = demo_test_metadata_path

    print("   Configuration updated for lightweight execution.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n2. Verifying 'library.utils'...")

    # Test RMSE Calculation
    # RMSE of [0, 0] vs [1, 1] should be 1.0
    y_true = np.zeros((5, 5))
    y_pred = np.ones((5, 5))
    rmse = utils.calculate_rmse(y_true, y_pred)
    assert abs(rmse - 1.0) < 1e-6, f"RMSE calculation failed. Expected 1.0, got {rmse}"
    print("   RMSE calculation verified.")

    # Test Image Loading
    # Load a sample image from the training set
    sample_img_path = os.path.join(config.INPUT_DIR, "train", "101.png")
    if os.path.exists(sample_img_path):
        img = utils.load_grayscale_image(sample_img_path)
        assert isinstance(img, np.ndarray), "Loaded image is not a numpy array"
        assert (
            img.dtype == np.float32
        ), f"Image dtype should be float32, got {img.dtype}"
        assert (
            0.0 <= img.min() and img.max() <= 1.0
        ), "Image pixels not normalized to [0, 1]"
        print(f"   Image loading verified. Shape: {img.shape}")
    else:
        print("   Skipping image load check (sample file not found).")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Preparation
    # -------------------------------------------------------------------------
    print("\n3. Verifying 'library.dataset'...")

    # Generate datasets (force regeneration with load_cached_data=False)
    # This uses the sparse stride defined in step 1
    train_ds, val_ds = dataset.prepare_datasets(load_cached_data=False)

    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."

    # Check a single sample
    x, y = train_ds[0]
    # Expected shape: (1, PATCH_SIZE, PATCH_SIZE)
    expected_shape = (1, config.PATCH_SIZE, config.PATCH_SIZE)
    assert (
        x.shape == expected_shape
    ), f"Patch shape mismatch. Expected {expected_shape}, got {x.shape}"
    assert (
        y.shape == expected_shape
    ), f"Target shape mismatch. Expected {expected_shape}, got {y.shape}"

    print(
        f"   Dataset created. Train samples: {len(train_ds)}, Val samples: {len(val_ds)}"
    )

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n4. Verifying 'library.model'...")

    # Instantiate model
    net = model.RDN(
        channel=config.IMG_CHANNELS,
        growth_rate=config.RDN_GROWTH_RATE,
        num_features=config.RDN_NUM_FEATURES,
        num_blocks=config.RDN_NUM_BLOCKS,
        num_layers=config.RDN_LAYERS_PER_BLOCK,
        kernel_size=config.RDN_KERNEL_SIZE,
    ).to(config.DEVICE)

    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 1, config.PATCH_SIZE, config.PATCH_SIZE).to(
        config.DEVICE
    )
    with torch.no_grad():
        output = net(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"

    print("   Model instantiated and forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Run Training
    # -------------------------------------------------------------------------
    print("\n5. Running Training Loop via 'library.train'...")

    # We use load_cached_data=True because we just generated the cache in step 3
    trained_model = train.train_model(load_cached_data=True)

    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("   Training completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Run Prediction
    # -------------------------------------------------------------------------
    print("\n6. Running Prediction via 'library.predict'...")

    # Generate predictions using the temp test metadata created in step 1
    predict.generate_predictions(
        model_path=config.MODEL_SAVE_PATH,
        metadata_path=config.TEST_METADATA_PATH,
        output_path=config.SUBMISSION_PATH,
        device=config.DEVICE,
    )

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"   Submission loaded. Rows: {len(df_sub)}")

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission missing required columns."

    # Check value range
    assert (
        df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
    ), "Submission values out of range [0, 1]."

    print("   Prediction pipeline verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    # Ensure reproducibility
    config.set_seed(42)
    run_demo()
