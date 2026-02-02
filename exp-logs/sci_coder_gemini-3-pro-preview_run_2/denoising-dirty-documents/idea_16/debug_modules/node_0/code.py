import os
import torch
import pandas as pd
import warnings

# Import components from the provided library
from library.config import (
    DEVICE,
    PATCH_SIZE,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    WORKING_DIR,
)
from library.utils import seed_everything
from library.dataset import DenoisingDataset
from library.model import ICResUNet
from library.train import run_training
from library.inference import predict_with_tta, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_dataset_loading():
    """
    Verifies that the dataset class loads data correctly for training and validation.
    """
    print("\n--- 1. Testing Dataset Loading ---")

    # Test Train Dataset (Patch-based)
    # We use a smaller samples_per_image for this quick check
    ds_train = DenoisingDataset(mode="train", samples_per_image=5)
    print(f"Train Dataset Length (Virtual): {len(ds_train)}")

    if len(ds_train) > 0:
        noisy, clean = ds_train[0]
        print(f"Train Sample Shape: Noisy {noisy.shape}, Clean {clean.shape}")

        # Assertions
        assert noisy.shape == (
            1,
            PATCH_SIZE,
            PATCH_SIZE,
        ), f"Expected (1, {PATCH_SIZE}, {PATCH_SIZE}), got {noisy.shape}"
        assert clean.shape == (
            1,
            PATCH_SIZE,
            PATCH_SIZE,
        ), f"Expected (1, {PATCH_SIZE}, {PATCH_SIZE}), got {clean.shape}"
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

    # Test Val Dataset (Full Image)
    ds_val = DenoisingDataset(mode="val")
    print(f"Val Dataset Length: {len(ds_val)}")

    if len(ds_val) > 0:
        noisy, clean, img_id = ds_val[0]
        print(f"Val Sample ID: {img_id}")
        print(f"Val Sample Shape: Noisy {noisy.shape}, Clean {clean.shape}")

        # Assertions
        assert (
            noisy.shape == clean.shape
        ), "Noisy and Clean images must have same dimensions"
        assert noisy.dim() == 3 and noisy.shape[0] == 1, "Image must be (1, H, W)"

    print("Dataset tests passed.")


def test_model_architecture():
    """
    Verifies that the model can be instantiated and process a dummy input.
    """
    print("\n--- 2. Testing Model Architecture ---")
    model = ICResUNet().to(DEVICE)
    model.eval()

    # Create dummy input: (Batch=2, Channels=1, Height, Width)
    dummy_input = torch.randn(2, 1, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == dummy_input.shape, "Output shape mismatch"
    print("Model architecture test passed.")


def test_training_loop():
    """
    Runs the training loop in debug mode for 1 epoch to verify pipeline integrity.
    """
    print("\n--- 3. Running Training Loop (Debug Mode) ---")

    # Run training
    # debug=True truncates the dataset to 10 images
    # epochs=1 ensures quick execution
    run_training(debug=True, epochs=1)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Training pipeline completed. Checkpoint verified at {checkpoint_path}")


def test_inference_logic():
    """
    Verifies the TTA inference logic on a single test sample.
    """
    print("\n--- 4. Testing Inference Logic ---")
    model = ICResUNet().to(DEVICE)
    model.eval()

    # Load a test sample
    ds_test = DenoisingDataset(mode="test")
    if len(ds_test) == 0:
        print("No test data found, skipping inference test.")
        return

    noisy, img_id = ds_test[0]
    noisy = noisy.to(DEVICE)  # Shape: (1, H, W)

    print(f"Testing inference on image {img_id} with shape {noisy.shape}")

    # Test TTA Prediction
    with torch.no_grad():
        prediction = predict_with_tta(model, noisy)

    print(f"Prediction Shape: {prediction.shape}")

    # Assertions
    assert prediction.shape == noisy.shape, "Prediction shape mismatch"
    assert (
        prediction.min() >= 0.0 and prediction.max() <= 1.0
    ), "Prediction values out of range [0, 1]"

    print("Inference logic test passed.")


def generate_full_submission():
    """
    Generates the final submission CSV using the trained model.
    """
    print("\n--- 5. Generating Submission ---")

    # This function loads 'best_model.pth' (created in step 3) and processes the test set
    generate_submission(checkpoint_name="best_model.pth")

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    # Validate CSV structure
    df = pd.read_csv(submission_path, nrows=5)
    print("Submission Head:")
    print(df.head())

    assert "id" in df.columns and "value" in df.columns, "Submission columns mismatch"
    assert len(df) > 0, "Submission file is empty"

    print(f"Submission generated successfully at {submission_path}")


def main():
    print(f"Running Demo on Device: {DEVICE}")
    seed_everything(42)

    # Execute steps sequentially
    test_dataset_loading()
    test_model_architecture()
    test_training_loop()
    test_inference_logic()
    generate_full_submission()

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
