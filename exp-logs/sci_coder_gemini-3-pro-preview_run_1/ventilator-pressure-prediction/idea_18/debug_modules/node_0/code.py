import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.dataset import prepare_data
from library.model import VentilatorModel
from library.trainer import Trainer, set_seed
from library.inference import generate_predictions


def main():
    print("=== Ventilator Pressure Prediction Demo ===")

    # 1. Configuration
    # Enable debug mode for fast execution (subsamples data, reduces epochs)
    config = Config(debug=True)
    print(f"Configuration loaded. Debug mode: {config.debug}")
    print(f"Working Directory: {config.WORKING_DIR}")

    # Clean working directory to ensure a fresh demonstration
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set global seeds for reproducibility
    set_seed(config.SEED)

    # 2. Data Preparation
    print("\n--- Data Preparation ---")
    # Load and process training data (Feature Engineering -> Scaling -> Reshaping)
    # load_cached_data=False forces the pipeline to run from scratch
    train_dataset = prepare_data(config, split="train", load_cached_data=False)
    val_dataset = prepare_data(config, split="val", load_cached_data=False)

    # Verification: Check dataset size and shapes
    # In debug mode, FeatureEngineer subsamples to 100 breaths
    # Each breath has 80 time steps.
    expected_breaths = 100
    expected_steps = 80

    print(f"Train Dataset Size: {len(train_dataset)}")
    assert (
        len(train_dataset) == expected_breaths
    ), f"Expected {expected_breaths} breaths in debug mode, got {len(train_dataset)}"

    # Check input shape: (80, n_features)
    sample_input = train_dataset[0]["input"]
    n_features = len(config.CONT_FEATURES) + len(config.BINARY_FEATURES)
    print(f"Sample Input Shape: {sample_input.shape}")
    assert sample_input.shape == (
        expected_steps,
        n_features,
    ), f"Expected input shape ({expected_steps}, {n_features}), got {sample_input.shape}"

    # Check target shape
    sample_target = train_dataset[0]["target"]
    print(f"Sample Target Shape: {sample_target.shape}")
    assert sample_target.shape == (
        expected_steps,
    ), f"Expected target shape ({expected_steps},), got {sample_target.shape}"

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = VentilatorModel(config)
    device = torch.device(config.DEVICE)
    model.to(device)

    # Verification: Dummy Forward Pass
    # Create a dummy batch of size 2
    dummy_input = torch.randn(2, 80, n_features).to(device)
    dummy_u_out = torch.zeros(2, 80).to(device)  # All inspiration phase

    with torch.no_grad():
        final_pred, aux_pred = model(dummy_input, u_out=dummy_u_out)

    print(f"Model Output Shape: {final_pred.shape}")
    assert final_pred.shape == (
        2,
        80,
        1,
    ), f"Expected output shape (2, 80, 1), got {final_pred.shape}"

    # 4. Training
    print("\n--- Training Loop ---")
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple debug run to avoid multiprocessing overhead
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize Trainer
    trainer = Trainer(config, model)

    # Run Training
    # Debug mode sets epochs to 2, so this should be quick
    trainer.fit(train_loader, val_loader)

    # Verification: Check if model checkpoint exists
    model_path = os.path.join(config.WORKING_DIR, "model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved!"
    print(f"Model successfully saved to {model_path}")

    # 5. Inference
    print("\n--- Inference ---")
    # Generate predictions on test set
    # This function handles loading the saved model and processing test data
    generate_predictions(config, load_cached_data=False)

    # Verification: Check submission file
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found!"

    # Validate submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["id", "pressure"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(sub_df.columns)}"

    # Check row count
    # In debug mode, test set is also subsampled to 100 breaths * 80 steps = 8000 rows
    expected_rows = 100 * 80
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in debug submission, got {len(sub_df)}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaN values!"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
