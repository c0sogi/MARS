import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import the provided library modules
from library import config, utils, model, data_loader, train


def main():
    print("=== Starting Demonstration Script ===")

    # =========================================================================
    # 1. Runtime Configuration Overrides
    # =========================================================================
    print("\n[Step 1] Configuring runtime environment for speed...")

    # Override config to run in debug mode (small data subset)
    config.DEBUG = True
    config.MAX_DEBUG_SAMPLES = 50  # Use only 50 samples

    # Override training parameters for a quick run
    config.MAX_EPOCHS = 1
    config.NUM_FOLDS = 1  # Only train fold 0
    config.BATCH_SIZE = 8
    config.PATIENCE = 1

    # Redirect output directories to a demo-specific folder
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.PROCESSED_DATA_PATH = os.path.join(config.CACHE_DIR, "processed_data.npz")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    utils.seed_everything(config.SEED)
    print("Configuration updated. Debug mode: ON")

    # =========================================================================
    # 2. Data Pipeline Verification
    # =========================================================================
    print("\n[Step 2] Verifying Data Processing Pipeline...")

    # Force processing of data (load_cached_data=False to ensure logic runs)
    # Note: In a real run, we might want to use cached data, but here we test the processor.
    X_train, y_train, inc_train, X_test, inc_test, ids_test = data_loader.process_data(
        load_cached_data=False
    )

    # Validation checks
    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Test Shape: {X_test.shape}")

    # Check Image Dimensions: (N, 3, 75, 75)
    assert X_train.ndim == 4
    assert X_train.shape[1] == 3
    assert X_train.shape[2] == 75
    assert X_train.shape[3] == 75

    # Check Label Dimensions
    assert len(y_train) == len(X_train)

    # Check Incidence Angle Dimensions
    assert len(inc_train) == len(X_train)

    print("Data pipeline verification passed.")

    # =========================================================================
    # 3. Model Architecture Verification
    # =========================================================================
    print("\n[Step 3] Verifying Model Architecture (GDPNet)...")

    # Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.GDPNet().to(device)

    # Create dummy input
    dummy_batch_size = 4
    dummy_img = torch.randn(dummy_batch_size, 3, 75, 75).to(device)
    dummy_ang = torch.randn(dummy_batch_size).to(device)

    # Forward pass
    net.eval()
    with torch.no_grad():
        output = net(dummy_img, dummy_ang)

    print(f"Model Output Shape: {output.shape}")

    # Validate output shape: (Batch, 1)
    assert output.shape == (
        dummy_batch_size,
        1,
    ), f"Expected {(dummy_batch_size, 1)}, got {output.shape}"

    print("Model architecture verification passed.")

    # =========================================================================
    # 4. Training Loop Execution
    # =========================================================================
    print("\n[Step 4] Executing Training Loop (Fold 0)...")

    # This function handles data loading, model init, and the training loop
    # It saves the model to config.WORKING_DIR
    train.train_fold(0)

    # Verify the model file was created
    expected_model_path = os.path.join(config.WORKING_DIR, "dpcnet_fold_0.pth")
    # Note: The provided code saves as "gdpnet_fold_{fold_index}.pth" in train.py
    # Let's check the train.py content provided in the prompt.
    # The provided train.py saves as: f"gdpnet_fold_{fold_index}.pth"
    expected_model_path = os.path.join(config.WORKING_DIR, "gdpnet_fold_0.pth")

    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model file at {expected_model_path}"
        )

    print(f"Training complete. Model saved to {expected_model_path}")

    # =========================================================================
    # 5. Inference and Submission Generation
    # =========================================================================
    print("\n[Step 5] Running Inference and Generating Submission...")

    # Load the trained model
    inference_model = model.GDPNet().to(device)
    inference_model = utils.load_checkpoint(
        inference_model, expected_model_path, device=device
    )
    inference_model.eval()

    # Get Test Loader
    test_loader, test_ids = data_loader.get_test_loader(load_cached_data=True)

    predictions = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            outputs = inference_model(images, angles)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Convert to numpy array
    predictions = np.array(predictions)

    # Validate predictions
    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save submission
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
