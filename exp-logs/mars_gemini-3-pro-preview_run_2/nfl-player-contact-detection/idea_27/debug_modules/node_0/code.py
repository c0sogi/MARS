import os
import torch
import pandas as pd
import numpy as np
import joblib

# Import from the provided library
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import get_dataloaders, get_test_loader
from library.train import train_model, get_vocab_sizes
from library.inference import optimize_threshold, generate_predictions
from library.model import SEARVN
import library.config as config


def main():
    # 1. Setup and Configuration
    print("=== Setting up Demonstration ===")
    seed_everything(config.SEED)

    # We use debug=True to limit the dataset size for speed (samples ~10k rows)
    DEBUG_MODE = True

    # 2. Data Processing Demonstration
    print("\n=== Step 1: Data Processing ===")
    # Initialize Processor
    processor = DataProcessor(debug=DEBUG_MODE)

    # Explicitly process train data to demonstrate the API and generate cache/encoders
    # load_cached_data=False forces it to process from raw CSVs for this demo
    print("Processing training data (generating features and fitting scalers)...")
    train_data_dict = processor.get_data(split="train", load_cached_data=False)

    # Validation: Check structure of returned data dictionary
    required_keys = ["X_kin", "X_vis", "X_cat", "y"]
    for key in required_keys:
        if key not in train_data_dict:
            raise AssertionError(f"Missing key '{key}' in processed data dictionary.")

    # Validation: Check Tensor Shapes
    # X_cat should have 4 columns: [P1_Pos, P2_Pos, P1_Team, P2_Team]
    assert (
        train_data_dict["X_cat"].shape[1] == 4
    ), "Categorical features should have 4 columns"
    # y should be 1D
    assert train_data_dict["y"].ndim == 1, "Target should be a 1D tensor"

    print(f"Processed Train Samples: {len(train_data_dict['y'])}")
    print("Encoders and Scalers saved successfully.")

    # Check if artifacts were saved
    if not os.path.exists(processor.encoders_path):
        raise FileNotFoundError("Encoders file was not saved.")
    if not os.path.exists(processor.scaler_path):
        raise FileNotFoundError("Scalers file was not saved.")

    # 3. Dataset and DataLoader Demonstration
    print("\n=== Step 2: Dataset and DataLoader ===")
    # get_dataloaders automatically handles processing (or loading cache) and wrapping in PyTorch Loaders
    train_loader, val_loader = get_dataloaders(debug=DEBUG_MODE, load_cached_data=True)

    # Fetch a single batch to verify DataLoader behavior
    batch = next(iter(train_loader))

    # Validation: Check Batch content
    assert "X_kin" in batch
    assert "contact_id" in batch
    # Check device placement readiness (should be CPU tensors initially)
    assert isinstance(batch["X_kin"], torch.Tensor)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print("DataLoader verification successful.")

    # 4. Model Instantiation Demonstration
    print("\n=== Step 3: Model Initialization ===")
    # We need vocab sizes to initialize the model (loaded from saved encoders)
    vocab_sizes = get_vocab_sizes()
    print(f"Vocabulary Sizes: {vocab_sizes}")

    model = SEARVN(vocab_sizes=vocab_sizes)

    # Validation: Test Forward Pass with the batch fetched earlier
    model.eval()
    with torch.no_grad():
        # Move inputs to appropriate types/device if needed (default is CPU here)
        logits = model(batch["X_kin"], batch["X_vis"], batch["X_cat"])

    # Output should be [Batch_Size, 1]
    assert logits.shape == (
        batch["y"].shape[0],
        1,
    ), f"Output shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n=== Step 4: Training Loop ===")
    # train_model is a high-level orchestrator provided in library.train
    # It handles the loop, validation, and saving the best model.
    # We rely on the config.EPOCHS (15) but since dataset is small (debug), it runs fast.
    best_model_path = train_model(debug=DEBUG_MODE, load_cached_data=True)

    print(f"Training complete. Best model saved to: {best_model_path}")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model file not found after training.")

    # 6. Inference and Submission Demonstration
    print("\n=== Step 5: Inference and Submission ===")

    # Optimize Threshold on Validation Set
    print("Optimizing decision threshold...")
    best_thresh, best_mcc = optimize_threshold(
        best_model_path, debug=DEBUG_MODE, load_cached_data=True
    )
    print(f"Optimal Threshold: {best_thresh:.4f} (Val MCC: {best_mcc:.4f})")

    # Ensure threshold is valid
    assert 0.0 < best_thresh < 1.0, "Threshold optimization returned invalid value."

    # Generate Predictions on Test Set
    print("Generating submission...")
    generate_predictions(
        best_model_path, best_thresh, debug=DEBUG_MODE, load_cached_data=True
    )

    # Validation: Check Submission File
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)

    # Check columns
    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns

    # Check values (should be binary)
    unique_preds = df_sub["contact"].unique()
    assert all(
        p in [0, 1] for p in unique_preds
    ), "Predictions must be binary (0 or 1)."

    print(f"Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
