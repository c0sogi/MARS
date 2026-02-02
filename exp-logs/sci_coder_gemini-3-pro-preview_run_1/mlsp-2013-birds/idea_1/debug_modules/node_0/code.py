import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.model import SimpleMLP
from library.data_loader import create_dataloaders
from library.trainer import ModelTrainer


def main():
    print("=== Starting Bird Species Prediction Pipeline Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid execution...")

    # Override default configuration for speed
    Config.DEBUG_SUBSET_SIZE = 50  # Limit training data to 50 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Use a small batch size
    Config.EARLY_STOPPING_PATIENCE = 1  # Minimal patience

    # Set seed for reproducibility
    Config.set_seed(42)

    # Create necessary output directories
    Config.create_directories()
    print("Configuration updated: DEBUG_SUBSET_SIZE=50, NUM_EPOCHS=2")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[Step 2] Creating DataLoaders...")

    # Force loading from scratch (load_cached_data=False) to verify processing logic
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Verify Train Loader
    features, labels, rec_ids = next(iter(train_loader))
    print(f"Train Batch Shapes -> Features: {features.shape}, Labels: {labels.shape}")

    assert (
        features.shape[1] == Config.INPUT_DIM
    ), f"Input feature dimension mismatch. Expected {Config.INPUT_DIM}, got {features.shape[1]}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Label dimension mismatch. Expected {Config.NUM_CLASSES}, got {labels.shape[1]}"
    assert features.shape[0] <= Config.BATCH_SIZE, "Batch size exceeds configuration."

    print("Data loading verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    device = Config.get_device()
    model = SimpleMLP(input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES).to(
        device
    )

    # Verify Forward Pass
    dummy_input = torch.randn(2, Config.INPUT_DIM).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES})"

    print("Model initialization verified successfully.")

    # ------------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------------
    print("\n[Step 4] Executing Training Loop...")

    # Initialize Trainer
    trainer = ModelTrainer(device=device)

    # Run Training
    # Note: The trainer prints loss metrics internally
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # Verify that the model state exists
    assert trainer.model is not None, "Trainer model should be initialized."
    print("Training loop executed successfully.")

    # ------------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Set...")

    predictions, pred_ids = trainer.predict(test_loader)

    # Verify Predictions
    # Test set size is fixed at 64 in the metadata
    expected_test_size = 64

    print(f"Predictions Shape: {predictions.shape}")

    assert predictions.shape == (
        expected_test_size,
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected ({expected_test_size}, {Config.NUM_CLASSES}), got {predictions.shape}"
    assert len(pred_ids) == expected_test_size, "Prediction IDs length mismatch."
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions contain values outside [0, 1] range."

    print("Inference verified successfully.")

    # ------------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------------
    print("\n[Step 6] Generating Submission File...")

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    trainer.generate_submission(predictions, pred_ids, output_path=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    expected_rows = expected_test_size * Config.NUM_CLASSES

    print(f"Submission Rows: {len(df_sub)}")
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert list(df_sub.columns) == ["Id", "Probability"], "Submission columns mismatch."

    # Verify ID formatting logic (rec_id * 100 + species_id)
    # Check the first entry
    first_rec_id = pred_ids[0]
    first_species_id = 0
    expected_id = first_rec_id * 100 + first_species_id
    actual_id = df_sub.iloc[0]["Id"]

    assert (
        actual_id == expected_id
    ), f"ID formatting error. Expected {expected_id}, got {actual_id}"

    print("Submission generation verified successfully.")
    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
