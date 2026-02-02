import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, TargetScaler
from library.data import get_dataloaders, process_structure
from library.model import SW_RA_CGN, train_one_epoch, validate, predict
from library.train import evaluate, generate_submission, compute_rmsle


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching for Demo Speed
    # -------------------------------------------------------------------------
    print("\n[1] Patching Configuration for fast execution...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update paths
    Config.TRAIN_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    Config.TARGET_SCALER_CACHE = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 20  # Only process 20 samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0
    Config.NUM_LAYERS = 2  # Reduce model size
    Config.HIDDEN_DIM = 32

    print("Config patched.")

    # -------------------------------------------------------------------------
    # 2. Testing Utils
    # -------------------------------------------------------------------------
    print("\n[2] Testing Utils (TargetScaler)...")
    scaler = TargetScaler()
    dummy_targets = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler.fit(dummy_targets)

    transformed = scaler.transform(dummy_targets)
    inverse = scaler.inverse_transform(transformed)

    print(f"Original: \n{dummy_targets}")
    print(f"Transformed: \n{transformed}")
    print(f"Inverse: \n{inverse}")

    assert np.allclose(dummy_targets, inverse), "TargetScaler inverse transform failed"
    print("TargetScaler logic verified.")

    # -------------------------------------------------------------------------
    # 3. Testing Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading...")
    # This will process raw data because cache doesn't exist in the new WORKING_DIR
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys}")
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch edge_index shape: {batch.edge_index.shape}")
    print(f"Batch y shape: {batch.y.shape}")

    assert batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert batch.y.shape[1] == 2, "Target should have 2 columns"
    print("Data loading verified.")

    # -------------------------------------------------------------------------
    # 4. Testing Model
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")
    device = torch.device("cpu")  # Use CPU for simple demo
    model = SW_RA_CGN(Config).to(device)

    # Forward pass
    batch = batch.to(device)
    output = model(batch)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        batch.num_graphs,
        2,
    ), f"Expected output shape {(batch.num_graphs, 2)}, got {output.shape}"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Testing Training Loop Components
    # -------------------------------------------------------------------------
    print("\n[5] Testing Training Components...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.6f}")
    assert train_loss >= 0, "Train loss should be non-negative"

    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.6f}")

    # Evaluate (includes RMSLE)
    avg_loss, rmsle_score = evaluate(model, val_loader, criterion, device, scaler)
    print(f"Eval Loss: {avg_loss:.6f}, RMSLE: {rmsle_score:.6f}")

    # Save checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("Training components verified.")

    # -------------------------------------------------------------------------
    # 6. Testing Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Testing Inference and Submission Generation...")

    # Predict
    preds, ids = predict(model, test_loader, device, scaler)
    print(f"Predictions shape: {preds.shape}")
    print(f"IDs shape: {ids.shape}")

    assert preds.shape[1] == 2, "Predictions should have 2 columns"
    assert len(preds) == len(ids), "Predictions and IDs length mismatch"

    # Generate submission file
    generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission file created successfully.")
        print(df_sub.head())
        assert df_sub.shape[1] == 3, "Submission should have 3 columns"
        assert list(df_sub.columns) == [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ]
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
