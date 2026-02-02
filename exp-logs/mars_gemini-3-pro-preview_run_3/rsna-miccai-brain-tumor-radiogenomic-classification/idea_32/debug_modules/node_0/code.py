import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from provided libraries
from library.utils import seed_everything, get_device
from library.data import load_processed_data, BraTSDataset
from library.model import HRLNNet
from library.train import train_model
import library.config as config


def run_demo():
    print("Starting Demonstration...")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define demo parameters to ensure quick execution
    DEMO_SAMPLES = 8
    DEMO_EPOCHS = 2
    DEMO_BATCH_SIZE = 2

    # Ensure working directory exists (handled by library, but good to be aware)
    # config.WORKING_DIR is defined in library/config.py

    # ==========================================
    # 2. Data Loading & Processing Verification
    # ==========================================
    print("\n[Demo] Testing Data Processing...")

    # Load a small subset of training data
    # We set load_cached_data=False to force the raw DICOM processing logic to run
    # This verifies the image loading, resizing, and normalization pipeline.
    X_train, y_train, ids_train = load_processed_data(
        split_name="train", load_cached_data=False, max_samples=DEMO_SAMPLES
    )

    print(f"Loaded Train Data Shape: {X_train.shape}")
    print(f"Loaded Train Labels Shape: {y_train.shape}")

    # Assertions to verify data integrity
    assert len(X_train) == DEMO_SAMPLES
    assert len(y_train) == DEMO_SAMPLES
    # Check dimensions: (Batch, Channels, Height, Width)
    assert (
        X_train.shape[1] == config.TOTAL_CHANNELS
    ), f"Expected {config.TOTAL_CHANNELS} channels"
    assert X_train.shape[2] == config.IMAGE_SIZE, f"Expected height {config.IMAGE_SIZE}"
    assert X_train.shape[3] == config.IMAGE_SIZE, f"Expected width {config.IMAGE_SIZE}"
    # Check for NaNs
    assert not np.isnan(X_train).any(), "Processed data contains NaNs"

    # Test Dataset Class instantiation
    dataset = BraTSDataset(X_train, y_train)
    sample_x, sample_y = dataset[0]

    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.shape == (
        config.TOTAL_CHANNELS,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    )
    print("Data Processing & Dataset Verification Passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[Demo] Testing Model Architecture...")

    model = HRLNNet().to(device)

    # Create a dummy batch matching the input dimensions
    dummy_input = torch.randn(
        DEMO_BATCH_SIZE, config.TOTAL_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(device)

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assert output shape is (Batch_Size,) for binary classification
    assert output.shape == (
        DEMO_BATCH_SIZE,
    ), f"Expected output shape ({DEMO_BATCH_SIZE},), got {output.shape}"
    print("Model Architecture Verification Passed.")

    # ==========================================
    # 4. Training Loop Verification
    # ==========================================
    print("\n[Demo] Testing Training Loop...")

    # Execute the training pipeline
    # We pass max_samples to limit the dataset size for the demo
    best_model_path = train_model(
        load_cached_data=False,  # Force processing
        num_epochs=DEMO_EPOCHS,
        batch_size=DEMO_BATCH_SIZE,
        learning_rate=1e-4,
        patience=2,  # Short patience
        max_samples=DEMO_SAMPLES,
    )

    # Verify model artifact creation
    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"Training finished. Model saved at: {best_model_path}")

    # ==========================================
    # 5. Inference & Submission Verification
    # ==========================================
    print("\n[Demo] Testing Inference & Submission Generation...")

    # Load a subset of Test Data
    X_test, _, ids_test = load_processed_data(
        split_name="test", load_cached_data=False, max_samples=DEMO_SAMPLES
    )

    test_dataset = BraTSDataset(X_test, None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=False
    )

    # Load the trained model
    model = HRLNNet().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    predictions = np.array(predictions)

    # Verify predictions
    assert len(predictions) == len(ids_test)
    assert np.all(predictions >= 0.0) and np.all(
        predictions <= 1.0
    ), "Predictions out of range [0, 1]"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": predictions})

    print("Sample Submission Rows:")
    print(submission_df.head())

    # Save to a demo submission file
    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_sub_path, index=False)
    assert os.path.exists(demo_sub_path)

    print("\n[Demo] All verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
