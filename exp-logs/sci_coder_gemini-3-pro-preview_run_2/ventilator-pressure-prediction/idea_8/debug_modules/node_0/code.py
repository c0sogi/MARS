import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import provided library components
from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.data_processing import prepare_datasets
from library.model import GIDBiLSTM
from library.engine import Trainer


def demo_pipeline():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> [1/6] Setting up Configuration...")

    # Override Config for a fast, isolated demonstration
    Config.WORKING_DIR = "./working/demo_execution_test"
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Use only 50 breaths for training/val to be fast
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.DEVICE = "cpu"  # Use CPU for deterministic, simple verification

    # Ensure working directory is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update derived paths in Config
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Initialize seeds
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print("\n>>> [2/6] Processing Data...")

    # Run data preparation (Debug mode = True, Load Cache = False to force processing)
    # Note: prepare_datasets processes full test set even in debug mode
    train_ds, val_ds, test_ds = prepare_datasets(debug=True, load_cached_data=False)

    # Validate Dataset Integrity
    print(
        f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}"
    )
    assert (
        len(train_ds) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} training breaths"
    assert (
        len(val_ds) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} validation breaths"

    # Validate Feature Shapes
    # Item structure: (X, u_out, y)
    sample_X, sample_u_out, sample_y = train_ds[0]

    # Expected shape: (80, INPUT_DIM)
    # Check Config.INPUT_DIM (defined as 18 in provided config)
    assert sample_X.shape == (80, 18), f"Feature shape mismatch. Got {sample_X.shape}"
    assert sample_u_out.shape == (80,), "u_out shape mismatch"
    assert sample_y.shape == (80,), "Target shape mismatch"

    print("Data processing verified.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> [3/6] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = GIDBiLSTM().to(device)

    # Create a dummy batch to verify forward pass
    # Shape: (Batch, Seq_Len, Input_Dim)
    dummy_batch_size = 4
    dummy_input = torch.randn(dummy_batch_size, 80, 18).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Expected Output: (Batch, Seq_Len)
    assert output.shape == (
        dummy_batch_size,
        80,
    ), f"Model output shape mismatch. Got {output.shape}"
    print("Model forward pass successful.")

    # ==========================================
    # 4. Loss Function Logic
    # ==========================================
    print("\n>>> [4/6] Verifying Weighted L1 Loss...")

    criterion = WeightedL1Loss()

    # Mock data
    preds = torch.tensor([10.0, 10.0])
    targets = torch.tensor([20.0, 20.0])
    # u_out: 0 = Inspiratory, 1 = Expiratory
    u_out = torch.tensor([0.0, 1.0])

    # Expected Calculation:
    # Item 1 (Insp): |10-20| * 1.0 = 10.0
    # Item 2 (Exp):  |10-20| * 0.1 = 1.0
    # Mean: (10.0 + 1.0) / 2 = 5.5

    loss = criterion(preds, targets, u_out)
    expected_loss = 5.5

    assert torch.isclose(
        loss, torch.tensor(expected_loss)
    ), f"Loss calculation error. Got {loss.item()}, expected {expected_loss}"
    print("Loss function logic verified.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n>>> [5/6] Executing Training Loop (1 Epoch)...")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, device, optimizer=optimizer, criterion=criterion)

    # Train
    train_loss = trainer.train_one_epoch(train_loader)
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Validate
    val_loss, val_mae = trainer.validate(val_loader)
    print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    # Save Checkpoint
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Checkpoint file not created"
    print("Checkpoint saved.")

    # ==========================================
    # 6. Inference Verification
    # ==========================================
    print("\n>>> [6/6] Verifying Inference...")

    # Use a subset of test data for speed
    test_subset = Subset(test_ds, indices=range(100))  # 100 breaths
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Reload model
    model_infer = GIDBiLSTM().to(device)
    model_infer.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    )
    trainer_infer = Trainer(model_infer, device)

    # Predict
    preds = trainer_infer.predict(test_loader)

    # Verify shape: 100 breaths * 80 steps = 8000 predictions
    expected_preds = 100 * 80
    assert (
        len(preds) == expected_preds
    ), f"Prediction count mismatch. Got {len(preds)}, expected {expected_preds}"
    print(f"Generated {len(preds)} predictions successfully.")

    print("\n>>> Demo Pipeline Completed Successfully.")


if __name__ == "__main__":
    demo_pipeline()
