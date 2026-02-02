import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data_processing import prepare_data
from library.dataset import VentilatorDataset
from library.model import VentilatorNet
from library.train import train_epoch, validate_epoch, inference, masked_mae_loss


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("=== 1. Setup & Configuration ===")

    # Override Config for a fast demonstration
    Config.EXP_ID = "demo_execution"
    Config.DEBUG = True  # Use small data subset
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Update paths based on new EXP_ID
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_ID)
    Config.SUBMISSION_DIR = "./submission"

    # Update cache paths manually since they were defined at class level in Config
    Config.CACHE_TRAIN_X = os.path.join(Config.WORKING_DIR, "train_x.npy")
    Config.CACHE_TRAIN_Y = os.path.join(Config.WORKING_DIR, "train_y.npy")
    Config.CACHE_VAL_X = os.path.join(Config.WORKING_DIR, "val_x.npy")
    Config.CACHE_VAL_Y = os.path.join(Config.WORKING_DIR, "val_y.npy")
    Config.CACHE_TEST_X = os.path.join(Config.WORKING_DIR, "test_x.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.CACHE_SCALER_CENTER = os.path.join(Config.WORKING_DIR, "scaler_center.npy")
    Config.CACHE_SCALER_SCALE = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean working directory to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Logic (Metric Calculation)
    # -------------------------------------------------------------------------
    print("\n=== 2. Verifying Metric Logic ===")

    # Create dummy data
    # Case: u_out=0 (Inspiration), error should count
    # Case: u_out=1 (Expiration), error should be ignored
    dummy_preds = torch.tensor([10.0, 20.0, 30.0])
    dummy_targets = torch.tensor([12.0, 20.0, 35.0])  # Errors: 2.0, 0.0, 5.0
    dummy_u_out = torch.tensor([0.0, 0.0, 1.0])  # Mask: Keep, Keep, Ignore

    mae = compute_metric(dummy_preds, dummy_targets, dummy_u_out)

    # Expected: (|10-12| + |20-20|) / 2 = (2 + 0) / 2 = 1.0
    # The 3rd element error (5.0) is ignored because u_out is 1.
    print(f"Computed MAE: {mae}")
    assert abs(mae - 1.0) < 1e-6, f"Metric calculation failed. Expected 1.0, got {mae}"
    print("Metric logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Processing
    # -------------------------------------------------------------------------
    print("\n=== 3. Running Data Processing ===")

    # This will load metadata, sample it (DEBUG=True), engineer features, scale, and cache
    train_x, train_y, val_x, val_y, test_x, test_ids = prepare_data(
        debug=True, load_cached_data=False
    )

    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Test X shape:  {test_x.shape}")

    # Assertions
    # Shape should be (N_breaths, 80, N_features)
    assert train_x.ndim == 3, "Train X should be 3D"
    assert train_x.shape[1] == 80, "Sequence length should be 80"
    assert (
        train_x.shape[0] == train_y.shape[0]
    ), "Mismatch between X and Y breath counts"
    assert train_y.shape[1] == 80, "Target sequence length should be 80"
    assert not np.isnan(train_x).any(), "NaNs found in training data"

    print("Data processing verified.")

    # -------------------------------------------------------------------------
    # 4. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n=== 4. Testing Dataset & DataLoader ===")

    train_dataset = VentilatorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch
    batch_x, batch_y, batch_u_out = next(iter(train_loader))

    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
    print(f"Batch u_out shape: {batch_u_out.shape}")

    # Assertions
    assert batch_x.shape == (Config.BATCH_SIZE, 80, train_x.shape[2])
    assert batch_y.shape == (Config.BATCH_SIZE, 80)
    assert batch_u_out.shape == (Config.BATCH_SIZE, 80)
    assert isinstance(batch_x, torch.Tensor)

    print("Dataset and DataLoader verified.")

    # -------------------------------------------------------------------------
    # 5. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n=== 5. Testing Model Architecture ===")

    input_dim = train_x.shape[2]
    model = VentilatorNet(input_dim=input_dim).to(device)

    # Move batch to device
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_u_out = batch_u_out.to(device)

    # Forward pass
    preds = model(batch_x)

    print(f"Predictions shape: {preds.shape}")

    # Assertions
    assert preds.shape == (Config.BATCH_SIZE, 80), "Model output shape mismatch"

    # Test Loss Function
    loss = masked_mae_loss(preds, batch_y, batch_u_out)
    print(f"Initial Batch Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Test Backward pass (Gradient check)
    loss.backward()
    # Check if gradients exist for a sample parameter
    param = next(model.parameters())
    assert param.grad is not None, "Gradients not computed"

    print("Model architecture and gradient flow verified.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n=== 6. Simulating Training Loop ===")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
    )

    # Run one epoch of training
    print("Running Train Epoch...")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Create Validation Loader
    val_dataset = VentilatorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run validation
    print("Running Validation Epoch...")
    val_loss, val_mae = validate_epoch(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    assert train_loss > 0
    assert val_loss > 0

    # Save model (simulating the checkpointing)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file not saved"

    print("Training loop functions verified.")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n=== 7. Testing Inference & Submission ===")

    # Create Test Loader
    test_dataset = VentilatorDataset(test_x, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Inference
    preds_matrix = inference(model, test_loader, device)
    print(f"Inference Output Shape: {preds_matrix.shape}")

    assert preds_matrix.shape == (
        test_x.shape[0],
        80,
    ), "Inference output shape mismatch"

    # Flatten and create submission
    preds_flat = preds_matrix.flatten()

    # Verify test_ids length matches predictions
    # test_ids is (N_breaths * 80,) because it comes from the raw dataframe ids
    # But in prepare_data, test_ids is just the array of IDs.
    # Let's verify the length match.
    assert len(test_ids) == len(
        preds_flat
    ), f"Length mismatch: IDs {len(test_ids)} vs Preds {len(preds_flat)}"

    submission = pd.DataFrame({"id": test_ids, "pressure": preds_flat})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission.head())

    print("\n=== Demo Execution Complete ===")


if __name__ == "__main__":
    main()
