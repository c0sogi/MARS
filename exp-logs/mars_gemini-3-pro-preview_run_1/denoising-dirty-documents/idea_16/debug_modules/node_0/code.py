import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import library modules
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.dataset as dataset_lib
import library.train_engine as train_engine
import library.inference_engine as inference_engine


def run_demo():
    print("=== Starting Denoising Pipeline Demo ===")

    # =========================================================================
    # 1. CONFIGURATION OVERRIDES FOR SPEED
    # =========================================================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Modify mutable config variables directly
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.TTA_ENABLED = False  # Disable TTA to speed up inference

    # Modify Stream Configs to run only one specific training job
    # We modify the dictionary contents so references in other modules see the change
    config.STREAM_A_CONFIG["seeds"] = [42]
    config.STREAM_B_CONFIG["seeds"] = []  # Skip Stream B for this demo

    # Set seed for reproducibility
    config.set_seed(42)

    print(f"Epochs set to: {config.EPOCHS}")
    print(f"Batch Size set to: {config.BATCH_SIZE}")
    print(f"Stream A Seeds: {config.STREAM_A_CONFIG['seeds']}")
    print(f"Stream B Seeds: {config.STREAM_B_CONFIG['seeds']}")

    # =========================================================================
    # 2. VERIFY UTILS
    # =========================================================================
    print("\n[2] Verifying Utility Functions...")

    # Load metadata to get a valid image path
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_df.iloc[0]
    sample_path = sample_row["noisy_image_path"]

    # Test load_image
    img = utils.load_image(sample_path)
    assert isinstance(img, np.ndarray), "Loaded image should be a numpy array"
    assert img.ndim == 2, "Image should be grayscale (2D)"
    assert (
        0.0 <= img.min() and img.max() <= 1.0
    ), "Pixels should be normalized to [0, 1]"
    print(f"Image loaded successfully. Shape: {img.shape}")

    # Test calculate_rmse
    rmse_val = utils.calculate_rmse([img], [img])
    assert rmse_val == 0.0, "RMSE between identical images should be 0.0"

    # Create a noisy version and check RMSE > 0
    noisy_img = img + 0.1
    rmse_noisy = utils.calculate_rmse([img], [noisy_img])
    assert rmse_noisy > 0.0, "RMSE should be positive for different images"
    print(f"RMSE check passed. Identical: {rmse_val}, Diff: {rmse_noisy:.4f}")

    # =========================================================================
    # 3. VERIFY DATASET
    # =========================================================================
    print("\n[3] Verifying Dataset Class...")

    # Create a small dataset for verification
    # We use a small subset of the dataframe
    subset_df = train_df.head(10)

    # Initialize dataset with augmentation enabled (Training Mode)
    ds = dataset_lib.DenoisingDataset(
        subset_df, img_size=(160, 160), augment=True, cache_name="demo_train_cache"
    )

    assert len(ds) == 10, "Dataset length mismatch"

    # Fetch one item
    noisy_t, clean_t, img_id = ds[0]

    # Check types and shapes
    assert isinstance(noisy_t, torch.Tensor), "Output should be a torch Tensor"
    assert isinstance(clean_t, torch.Tensor), "Output should be a torch Tensor"
    assert noisy_t.shape == (
        1,
        160,
        160,
    ), f"Expected shape (1, 160, 160), got {noisy_t.shape}"
    assert clean_t.shape == (
        1,
        160,
        160,
    ), f"Expected shape (1, 160, 160), got {clean_t.shape}"

    print("Dataset verification passed.")

    # =========================================================================
    # 4. VERIFY MODEL
    # =========================================================================
    print("\n[4] Verifying Model Architecture...")

    device = utils.get_device()
    model = model_lib.ResolutionPreservedUNet().to(device)

    # Create dummy input: Batch=2, Channel=1, H=160, W=160
    dummy_input = torch.randn(2, 1, 160, 160).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape {output.shape} does not match input {dummy_input.shape}"

    print("Model forward pass successful. Output shape matches input.")

    # =========================================================================
    # 5. RUN TRAINING DEMO
    # =========================================================================
    print("\n[5] Executing Training Pipeline (Stream A, Seed 42)...")

    # We use debug_max_samples to limit the training data to 20 samples
    # This ensures the epoch finishes almost instantly
    best_rmse = train_engine.train_model(
        stream_config=config.STREAM_A_CONFIG,
        seed_index=0,  # Use the first (and only) seed in our modified config: 42
        debug_max_samples=20,
    )

    # Verify model file was saved
    expected_model_path = os.path.join(
        config.WORKING_DIR, "StreamA_Context_seed_42.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"

    print(f"Training demo complete. Model saved to {expected_model_path}")

    # =========================================================================
    # 6. RUN INFERENCE DEMO
    # =========================================================================
    print("\n[6] Executing Inference Pipeline...")

    # We limit inference to 5 test images for speed
    inference_engine.predict_test_set(debug_max_samples=5)

    # Verify submission file
    submission_path = config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check columns
    assert (
        "id" in sub_df.columns and "value" in sub_df.columns
    ), "Submission missing required columns"

    # Check value range (should be roughly 0-1, though floats are allowed)
    # Since we save floats, let's just check they are numeric
    assert pd.api.types.is_numeric_dtype(
        sub_df["value"]
    ), "Value column should be numeric"

    print("Inference demo complete. Submission verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        raise e
