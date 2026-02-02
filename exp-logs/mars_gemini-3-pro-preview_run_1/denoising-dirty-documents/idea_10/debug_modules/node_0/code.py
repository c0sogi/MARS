import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import get_config, WORKING_DIR, SUBMISSION_FILE_PATH, DEVICE
from library.utils import set_seed, invert_intensity, revert_intensity
from library.models import get_context_specialist, get_texture_specialist
from library.train import train_instance
from library.inference import generate_submission


def test_utilities():
    """
    Verifies the correctness of utility functions, specifically intensity inversion.
    """
    print("\n--- Testing Utilities ---")

    # Test Data
    original = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    # 1. Test Invert Intensity (0 -> 1, 1 -> 0)
    inverted = invert_intensity(original)
    expected_inverted = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    np.testing.assert_allclose(
        inverted, expected_inverted, atol=1e-6, err_msg="Invert intensity failed"
    )
    print("✓ invert_intensity passed")

    # 2. Test Revert Intensity (should be symmetric to invert)
    reverted = revert_intensity(inverted)
    np.testing.assert_allclose(
        reverted, original, atol=1e-6, err_msg="Revert intensity failed"
    )
    print("✓ revert_intensity passed")


def test_model_architecture():
    """
    Instantiates models and runs a dummy forward pass to check dimensions.
    """
    print("\n--- Testing Model Architectures ---")

    batch_size = 2
    channels = 1
    size = 128
    dummy_input = torch.randn(batch_size, channels, size, size).to(DEVICE)

    # 1. Test Context Specialist (Stream A)
    model_a = get_context_specialist().to(DEVICE)
    output_a = model_a(dummy_input)

    assert output_a.shape == (
        batch_size,
        channels,
        size,
        size,
    ), f"Context Specialist output shape mismatch. Expected {(batch_size, channels, size, size)}, got {output_a.shape}"
    print("✓ Context Specialist instantiation and forward pass passed")

    # 2. Test Texture Specialist (Stream B)
    model_b = get_texture_specialist().to(DEVICE)
    output_b = model_b(dummy_input)

    assert output_b.shape == (
        batch_size,
        channels,
        size,
        size,
    ), f"Texture Specialist output shape mismatch. Expected {(batch_size, channels, size, size)}, got {output_b.shape}"
    print("✓ Texture Specialist instantiation and forward pass passed")


def run_training_demo():
    """
    Runs the training loop in debug mode for one instance of each stream.
    Debug mode reduces epochs to 2 and uses a subset of data if configured.
    """
    print("\n--- Running Training Demo (Debug Mode) ---")

    # Ensure working directory exists (handled by config, but good to be safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Train one instance of Context Specialist (ID 0)
    # This will save 'context_model_0.pth' to WORKING_DIR
    print("Training Context Specialist (ID 0)...")
    train_instance(stream_type="context", instance_id=0, debug=True)

    expected_path_a = os.path.join(WORKING_DIR, "context_model_0.pth")
    if not os.path.exists(expected_path_a):
        raise FileNotFoundError(f"Training failed: {expected_path_a} was not created.")
    print("✓ Context Specialist trained and saved.")

    # Train one instance of Texture Specialist (ID 0)
    # This will save 'texture_model_0.pth' to WORKING_DIR
    print("Training Texture Specialist (ID 0)...")
    train_instance(stream_type="texture", instance_id=0, debug=True)

    expected_path_b = os.path.join(WORKING_DIR, "texture_model_0.pth")
    if not os.path.exists(expected_path_b):
        raise FileNotFoundError(f"Training failed: {expected_path_b} was not created.")
    print("✓ Texture Specialist trained and saved.")


def run_inference_demo():
    """
    Runs the inference pipeline in debug mode.
    This loads the models trained above and generates a submission file.
    """
    print("\n--- Running Inference Demo (Debug Mode) ---")

    # Generate submission
    # Debug mode processes only 5 images from the test set
    generate_submission(debug=True)

    if not os.path.exists(SUBMISSION_FILE_PATH):
        raise FileNotFoundError(
            f"Inference failed: {SUBMISSION_FILE_PATH} was not created."
        )

    print("✓ Submission file generated.")


def verify_submission_file():
    """
    Validates the format and content of the generated submission file.
    """
    print("\n--- Verifying Submission File ---")

    df = pd.read_csv(SUBMISSION_FILE_PATH)

    # Check columns
    expected_cols = ["id", "value"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(
            f"Submission file missing required columns. Found: {df.columns}"
        )

    # Check if not empty
    if len(df) == 0:
        raise ValueError("Submission file is empty.")

    # Check value range [0, 1]
    min_val = df["value"].min()
    max_val = df["value"].max()

    if min_val < 0 or max_val > 1:
        raise ValueError(
            f"Pixel values out of range [0, 1]. Range found: [{min_val}, {max_val}]"
        )

    # Check ID format (e.g., '110_1_1')
    sample_id = str(df.iloc[0]["id"])
    parts = sample_id.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"ID format incorrect. Expected 'img_row_col', got '{sample_id}'"
        )

    print(f"✓ Submission file format verified. Rows: {len(df)}")
    print(f"  Sample ID: {sample_id}, Value: {df.iloc[0]['value']}")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)

    try:
        # 2. Unit Tests
        test_utilities()
        test_model_architecture()

        # 3. Integration Tests (Train & Inference)
        # Note: This uses the provided library functions which handle data loading,
        # caching, and model saving/loading internally.
        run_training_demo()
        run_inference_demo()

        # 4. Validation
        verify_submission_file()

        print("\nAll demonstration steps completed successfully.")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
