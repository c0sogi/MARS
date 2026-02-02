import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library import utils, data, model, engine, layers


def main():
    print("============================================================")
    print("      Cactus Identification Library Demonstration           ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # ------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Override Config values to run a small, fast experiment
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images for train/val/test
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.SEEDS = [42]  # Run only one seed
    Config.WORKING_DIR = "./working/demo_execution"  # Isolate demo outputs
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Re-run setup to create the new directories
    Config.setup()

    # Set seed for reproducibility
    utils.set_seed(42)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # ------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading and Caching...")

    # Clean up any existing cache in the demo dir to ensure fresh load
    for f in os.listdir(Config.WORKING_DIR):
        if f.endswith(".npy"):
            os.remove(os.path.join(Config.WORKING_DIR, f))

    batch_size = 8
    train_loader, train_ids = data.get_dataloader(
        "train", batch_size=batch_size, shuffle=True
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (batch_size, 3, 32, 32), "Incorrect image batch shape"
    assert labels.shape == (batch_size,), "Incorrect label batch shape"
    assert (
        images.dtype == torch.float32
    ), "Images should be FloatTensor (after ToTensorV2)"

    # Verify cache creation
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "train_images.npy")
    ), "Train images cache not created"
    print("Data loading and caching verified successfully.")

    # ------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture (CustomWideResNeSt)...")

    net = model.CustomWideResNeSt()
    net.to(Config.DEVICE)
    net.eval()

    # Count parameters
    num_params = utils.count_parameters(net)
    print(f"Model Parameters: {num_params:,}")

    # Forward pass check
    dummy_input = torch.randn(2, 3, 32, 32).to(Config.DEVICE)
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 1), "Model output shape mismatch (expected [Batch, 1])"
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("Model architecture verified successfully.")

    # ------------------------------------------------------------------
    # 4. Training Engine Demonstration
    # ------------------------------------------------------------------
    print("\n[Step 4] Demonstrating Training Loop (1 Epoch, Seed 42)...")

    # This function runs the training loop, validation, and saves the best model
    best_model_path = engine.train_seed(42, Config.DEVICE)

    # Assertions
    assert os.path.exists(
        best_model_path
    ), f"Model file was not saved at {best_model_path}"
    print(f"Training complete. Model saved to: {best_model_path}")

    # ------------------------------------------------------------------
    # 5. Inference and Submission Verification
    # ------------------------------------------------------------------
    print("\n[Step 5] Demonstrating Inference (TTA) and Submission Generation...")

    # Load Test Data
    test_loader, test_ids = data.get_dataloader(
        "test", batch_size=batch_size, shuffle=False
    )

    # Load the trained model
    inference_model = model.CustomWideResNeSt()
    inference_model.load_state_dict(
        torch.load(best_model_path, map_location=Config.DEVICE)
    )
    inference_model.to(Config.DEVICE)

    # Run Prediction with Test Time Augmentation
    predictions = engine.predict_with_tta(inference_model, test_loader, Config.DEVICE)

    print(f"Number of Test IDs: {len(test_ids)}")
    print(f"Number of Predictions: {len(predictions)}")

    # Assertions
    assert len(predictions) == len(
        test_ids
    ), "Mismatch between test IDs and predictions count"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Probabilities must be in [0, 1]"

    # Save Submission
    utils.save_submission(test_ids, predictions, save_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == ["id", "has_cactus"], "Incorrect submission columns"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"

    print("\nInference and submission verified successfully.")

    print("\n============================================================")
    print("      Demonstration Completed Successfully                  ")
    print("============================================================")


if __name__ == "__main__":
    main()
