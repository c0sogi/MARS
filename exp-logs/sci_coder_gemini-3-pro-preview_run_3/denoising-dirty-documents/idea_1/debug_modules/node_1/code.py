import os
import numpy as np
import pandas as pd
import cv2
import shutil

# Import from the provided library files
from library.config import (
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SEED,
    PATCH_SIZE,
    INPUT_DIR,
)
from library.data_loader import extract_patch_data
from library.model import LearnedLinearFilter
from library.utils import load_normalized_image, calculate_rmse, format_submission


def run_demonstration():
    print("Starting library demonstration...")

    # 1. Set Random Seed for Reproducibility
    np.random.seed(SEED)

    # Define a temporary output directory for this demo to avoid cluttering the main working dir
    # or to ensure we are testing file creation fresh.
    DEMO_OUTPUT_DIR = os.path.join(WORKING_DIR, "demo_output")
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # DEMO 1: Data Loading & Patch Extraction
    # =========================================================================
    print("\n[Demo 1] Extracting patches from training data...")

    # We use a small number of samples (e.g., 5000) to ensure the demo runs quickly.
    # We set load_cached_data=False to verify the extraction logic works.
    demo_num_samples = 5000

    X, y = extract_patch_data(
        metadata_path=TRAIN_METADATA_PATH,
        patch_size=PATCH_SIZE,
        num_samples=demo_num_samples,
        load_cached_data=False,
    )

    # Validation
    print(f"Extracted X shape: {X.shape}")
    print(f"Extracted y shape: {y.shape}")

    expected_features = PATCH_SIZE * PATCH_SIZE
    assert X.shape == (
        demo_num_samples,
        expected_features,
    ), f"Expected X shape ({demo_num_samples}, {expected_features}), got {X.shape}"
    assert y.shape == (
        demo_num_samples,
    ), f"Expected y shape ({demo_num_samples},), got {y.shape}"
    assert X.dtype == np.float32, "X should be float32"
    assert y.dtype == np.float32, "y should be float32"

    print("Data extraction verified successfully.")

    # =========================================================================
    # DEMO 2: Model Training
    # =========================================================================
    print("\n[Demo 2] Training LearnedLinearFilter...")

    model = LearnedLinearFilter(patch_size=PATCH_SIZE, alpha=1.0)
    model.fit(X, y)

    # Validation
    assert model.kernel is not None, "Model kernel should not be None after fitting"
    assert model.kernel.shape == (
        PATCH_SIZE,
        PATCH_SIZE,
    ), f"Kernel shape mismatch. Expected ({PATCH_SIZE}, {PATCH_SIZE}), got {model.kernel.shape}"
    assert isinstance(model.bias, float), "Bias should be a float"

    # Check if weights were saved to disk as per model.fit() implementation
    weights_path = os.path.join(WORKING_DIR, "model_weights.npy")
    bias_path = os.path.join(WORKING_DIR, "model_bias.npy")
    assert os.path.exists(weights_path), "Model weights file was not created"
    assert os.path.exists(bias_path), "Model bias file was not created"

    print("Model training and weight saving verified successfully.")

    # =========================================================================
    # DEMO 3: Evaluation (RMSE Calculation)
    # =========================================================================
    print("\n[Demo 3] Evaluating model on training subset...")

    # We reuse the training data here just to demonstrate the evaluate function
    # In a real scenario, this should be a validation set.
    rmse = model.evaluate(X[:100], y[:100])

    # Validation
    assert isinstance(rmse, float), "RMSE should be a float"
    assert rmse >= 0, "RMSE cannot be negative"

    # Verify utility function calculate_rmse separately
    dummy_true = np.array([0.0, 1.0, 0.5])
    dummy_pred = np.array([0.0, 1.0, 0.5])
    calc_rmse = calculate_rmse(dummy_true, dummy_pred)
    assert calc_rmse == 0.0, "RMSE of identical arrays should be 0.0"

    print(f"Evaluation verified. RMSE: {rmse:.4f}")

    # =========================================================================
    # DEMO 4: Inference on a Single Image
    # =========================================================================
    print("\n[Demo 4] Running inference on a sample image...")

    # Pick a sample image from the training set metadata to test inference
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    input_rel_path = sample_row["input_path"]
    input_full_path = os.path.join(INPUT_DIR, input_rel_path)

    # Load image using utility
    img_in = load_normalized_image(input_full_path)

    # Predict
    img_denoised = model.predict(img_in)

    # Validation
    assert (
        img_denoised.shape == img_in.shape
    ), f"Output shape {img_denoised.shape} does not match input shape {img_in.shape}"
    assert (
        img_denoised.min() >= 0.0 and img_denoised.max() <= 1.0
    ), "Denoised image values must be clipped to [0, 1]"
    assert (
        img_denoised.dtype == np.float32 or img_denoised.dtype == np.float64
    ), "Output image should be floating point"

    print("Inference verified successfully.")

    # =========================================================================
    # DEMO 5: Submission Formatting
    # =========================================================================
    print("\n[Demo 5] Formatting submission file...")

    # Create a dictionary mimicking the structure required by format_submission
    # We use the filename from the metadata (e.g., "101.png")
    sample_filename = sample_row["image_id"]
    predictions_dict = {sample_filename: img_denoised}

    submission_output_path = os.path.join(DEMO_OUTPUT_DIR, "demo_submission.csv")

    format_submission(predictions_dict, submission_output_path)

    # Validation
    assert os.path.exists(submission_output_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_output_path)
    print(f"Submission file created with {len(df_sub)} rows.")

    # Check columns
    assert list(df_sub.columns) == ["id", "value"], "Submission columns mismatch"

    # Check ID format (image_row_col)
    # img_denoised shape is (H, W). Total rows should be H * W.
    h, w = img_denoised.shape
    expected_rows = h * w
    assert (
        len(df_sub) == expected_rows
    ), f"Row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check first ID format
    first_id = df_sub.iloc[0]["id"]
    # Expected format: "101_1_1" (assuming filename is 101.png)
    base_id = os.path.splitext(sample_filename)[0]
    assert first_id.startswith(
        f"{base_id}_"
    ), f"ID {first_id} does not start with image ID {base_id}"

    print("Submission formatting verified successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demonstration()
