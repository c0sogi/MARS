import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, cleanup_cache
from library.features import FeatureEngineer
from library.data import VentilatorDataset, prepare_datasets
from library.model import PITHNet
from library.train import run_training, masked_mae_loss


def main():
    print("=== Starting Library Usage Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True  # Uses only 100 breaths per split
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist (Config.setup() was called on import, but we changed paths)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Clean up any previous demo runs
    cleanup_cache(Config.WORKING_DIR)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------------------------
    print("\n[2] Demonstrating FeatureEngineer...")

    fe = FeatureEngineer()

    # Process training data (this triggers scaling fitting)
    # load_cached_data=False forces processing from raw CSVs
    x_train, y_train, u_out_train, ids_train = fe.get_data(
        "train", load_cached_data=False
    )

    print(
        f"    Train Data Shapes: X={x_train.shape}, Y={y_train.shape}, u_out={u_out_train.shape}"
    )

    # Verification
    # Shape should be (N_breaths, 80, N_features)
    assert x_train.ndim == 3, "Input X should be 3-dimensional"
    assert x_train.shape[1] == 80, "Sequence length must be 80"
    assert (
        x_train.shape[2] == Config.INPUT_DIM
    ), f"Feature dim must match Config ({Config.INPUT_DIM})"
    assert y_train.shape == (x_train.shape[0], 80), "Target Y shape mismatch"
    assert u_out_train.shape == (x_train.shape[0], 80), "u_out shape mismatch"

    # Check scaling (RobustScaler centers around median, so values shouldn't be massive)
    assert (
        np.abs(np.median(x_train)).mean() < 10
    ), "Features do not appear to be scaled properly"

    # ------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # ------------------------------------------------------------------------
    print("\n[3] Demonstrating VentilatorDataset & prepare_datasets...")

    # Use the high-level function to get loaders
    # Since we just processed 'train' above, it will be cached.
    # 'val' and 'test' will be processed now.
    train_loader, val_loader, test_loader, test_ids = prepare_datasets(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch to verify
    batch = next(iter(train_loader))
    x_batch, y_batch, u_out_batch = batch["x"], batch["y"], batch["u_out"]

    print(f"    Batch Shapes: X={x_batch.shape}, Y={y_batch.shape}")

    # Verification
    assert x_batch.shape == (Config.BATCH_SIZE, 80, Config.INPUT_DIM)
    assert y_batch.shape == (Config.BATCH_SIZE, 80)
    assert isinstance(x_batch, torch.Tensor), "DataLoader should yield Tensors"
    assert x_batch.dtype == torch.float32, "Input tensor should be float32"

    # ------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating PITHNet Model...")

    model = PITHNet().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    u_out_batch = u_out_batch.to(device)

    # Forward pass
    preds = model(x_batch)

    print(f"    Prediction Shape: {preds.shape}")

    # Verification
    assert preds.shape == (Config.BATCH_SIZE, 80), "Prediction shape mismatch"

    # Loss Calculation
    loss = masked_mae_loss(preds, y_batch, u_out_batch)
    print(f"    Initial Loss (Random Weights): {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() >= 0, "MAE loss must be non-negative"

    # ------------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (run_training)...")

    # Run training for limited epochs (Config.EPOCHS=2)
    best_val_loss = run_training(epochs=Config.EPOCHS, load_cached_data=True)

    print(f"    Training finished. Best Val Loss: {best_val_loss:.4f}")

    # Verify model checkpoint creation
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved"

    # ------------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[6] Demonstrating Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            # Forward
            preds = model(x)
            # Flatten predictions (N, 80) -> (N*80,)
            all_preds.append(preds.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)

    # Flatten IDs for submission
    # test_ids is (N_breaths, 80)
    flat_ids = test_ids.flatten()

    print(f"    Total Predictions: {len(all_preds)}")
    print(f"    Total IDs: {len(flat_ids)}")

    # Verification
    assert len(all_preds) == len(flat_ids), "Mismatch between IDs and predictions"

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": flat_ids, "pressure": all_preds})

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Final check of the output file
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert saved_df.shape == (len(flat_ids), 2)
    assert list(saved_df.columns) == ["id", "pressure"]

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
