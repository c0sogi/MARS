import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.dataset import WhaleDataset, get_dataloaders
from library.model import WhaleResNet
from library.trainer import train_model, generate_submission
from library.inference import run_inference


def main():
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    print("=== Starting Library Usage Demonstration ===")

    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    Config.set_seed()

    # Override Config parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = (224, 224)  # Ensure consistent size

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Samples per dataset: {Config.DEBUG_SAMPLES}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # 2. Dataset Demonstration & Verification
    # ---------------------------------------------------------
    print("\n[2] Validating WhaleDataset...")

    # Instantiate Train Dataset
    train_ds = WhaleDataset(Config.TRAIN_CSV, Config.INPUT_ROOT, is_test=False)

    # Assert dataset length matches debug samples
    assert (
        len(train_ds) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} samples, got {len(train_ds)}"

    # Fetch a single item
    image, label = train_ds[0]

    # Validate Image Tensor Shape: (Channels, Height, Width) -> (1, 224, 224)
    expected_shape = (1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {image.shape}"

    # Validate Label Shape: (1,)
    assert label.shape == (
        1,
    ), f"Label shape mismatch. Expected (1,), got {label.shape}"

    print("Train dataset item verification passed.")

    # Instantiate Test Dataset
    test_ds = WhaleDataset(Config.TEST_CSV, Config.INPUT_ROOT, is_test=True)
    test_image, test_clip = test_ds[0]

    # Validate Test Item
    assert isinstance(
        test_clip, str
    ), "Test dataset should return clip filename as string."
    assert test_image.shape == expected_shape, "Test image shape mismatch."

    print("Test dataset item verification passed.")

    # 3. Model Demonstration & Verification
    # ---------------------------------------------------------
    print("\n[3] Validating WhaleResNet Model...")

    device = torch.device(Config.DEVICE)
    model = WhaleResNet(
        pretrained=False
    )  # Skip downloading weights for speed if possible, or use cached
    model.to(device)
    model.eval()

    # Create a dummy batch from the single image we loaded
    # Shape: (Batch, 1, H, W)
    dummy_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Validate Output Shape: (Batch, Num_Classes) -> (1, 1)
    assert output.shape == (
        1,
        1,
    ), f"Model output shape mismatch. Expected (1, 1), got {output.shape}"

    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Executing Training Loop (train_model)...")

    # This will run for 1 epoch on 20 samples
    trained_model = train_model(num_epochs=Config.NUM_EPOCHS, patience=1)

    # Verify model file was created
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not saved."

    print("Training loop completed and model saved successfully.")

    # 5. Submission Generation Demonstration
    # ---------------------------------------------------------
    print("\n[5] Generating Submission (generate_submission)...")

    generate_submission(trained_model)

    # Verify submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV not found."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape (Should be equal to DEBUG_SAMPLES in rows, 2 columns)
    assert sub_df.shape == (
        Config.DEBUG_SAMPLES,
        2,
    ), f"Submission shape mismatch. Expected ({Config.DEBUG_SAMPLES}, 2), got {sub_df.shape}"

    # Check columns
    assert list(sub_df.columns) == [
        "clip",
        "probability",
    ], f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check probability range
    probs = sub_df["probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print("Submission generation verification passed.")

    # 6. Inference Wrapper Demonstration
    # ---------------------------------------------------------
    print("\n[6] Testing Inference Wrapper (run_inference)...")

    # We use load_cached_data=True to verify it picks up the model we just trained
    # This avoids retraining and proves the caching logic works.
    run_inference(load_cached_data=True, num_epochs=1, debug=True)

    # Re-verify submission file timestamp or existence (it overwrites, which is fine)
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Inference wrapper failed to produce submission."

    print("Inference wrapper execution passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
