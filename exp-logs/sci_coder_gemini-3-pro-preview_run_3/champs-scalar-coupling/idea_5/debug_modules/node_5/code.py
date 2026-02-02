import torch
import pandas as pd
import numpy as np
import os
import shutil
import sys

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders, MoleculeDataset
from library.model import DualGraphGNN
from library.engine import Trainer
from library.utils import TargetScaler


def run_demo():
    print("Starting Scalar Coupling Prediction Demo...")

    # ==========================================
    # 1. Configure for Speed (Demo Mode)
    # ==========================================
    # Override Config parameters to ensure the script runs quickly
    print("Configuring for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset of molecules
    Config.MAX_EPOCHS = 2  # Minimal epochs to prove training loop works
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.NUM_INTERACTION_LAYERS = 2

    # Setup environment (seeds, directories)
    Config.setup_environment()

    # Clean up previous debug cache if exists to ensure fresh run
    if os.path.exists(Config.PROCESSED_DATA_DIR):
        for f in os.listdir(Config.PROCESSED_DATA_DIR):
            if "debug" in f:
                os.remove(os.path.join(Config.PROCESSED_DATA_DIR, f))

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n--- Data Loading ---")
    # This will process the raw data into graph structures (Atom Graph + Line Graph)
    # and cache them as .pt files.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Inspect a single batch
    batch = next(iter(train_loader))
    print(f"Batch loaded successfully.")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Num atoms: {batch.num_atoms}")
    print(f"  Node features (x) shape: {batch.x.shape}")
    print(f"  Edge index shape: {batch.edge_index.shape}")
    print(f"  Line edge index shape: {batch.line_edge_index.shape}")
    print(f"  Coupling targets (y_coupling) shape: {batch.y_coupling.shape}")

    # Verification: Check graph connectivity logic
    assert batch.edge_index.shape[0] == 2, "Edge index must have 2 rows (src, dst)"
    assert (
        batch.x.shape[0] == batch.num_atoms.sum()
    ), "Node features must match number of atoms"
    assert (
        batch.y_coupling.shape[0] == batch.type_coupling.shape[0]
    ), "Targets and types must match"

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- Model Initialization & Forward Pass ---")
    device = Config.get_device()
    model = DualGraphGNN().to(device)
    batch = batch.to(device)

    # Run forward pass
    pred_coupling, pred_shielding, pred_charge = model(batch)

    print("Forward pass complete.")
    print(f"  Coupling Pred Shape: {pred_coupling.shape}")
    print(f"  Shielding Pred Shape: {pred_shielding.shape}")
    print(f"  Charge Pred Shape: {pred_charge.shape}")

    # Verification: Output shapes
    # Coupling prediction should match the number of coupling pairs in the batch
    assert (
        pred_coupling.shape == batch.y_coupling.shape
    ), f"Coupling prediction shape mismatch: {pred_coupling.shape} vs {batch.y_coupling.shape}"

    # Shielding prediction: (Num Atoms, 9 tensor components)
    assert pred_shielding.shape == (
        batch.x.shape[0],
        9,
    ), "Shielding prediction shape mismatch"

    # Charge prediction: (Num Atoms,)
    assert pred_charge.shape == (batch.x.shape[0],), "Charge prediction shape mismatch"

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n--- Training Loop ---")
    # Initialize Trainer
    trainer = Trainer()

    # Verify Trainer initialized the model and scaler correctly
    assert isinstance(trainer.model, DualGraphGNN)
    assert isinstance(trainer.scaler, TargetScaler)

    # Run Training (Fit)
    # This runs for Config.MAX_EPOCHS (set to 2 above)
    trainer.fit(train_loader, val_loader)

    # Verify model artifact was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"
    print("Training loop completed and model saved.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n--- Inference ---")
    # Generate predictions on test set
    trainer.predict(test_loader)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Check format
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    # Check if IDs match the test loader subset
    # Since we used a random subset in debug mode, we just check consistency within the generated file
    assert df_sub["id"].is_monotonic_increasing, "Submission IDs should be sorted"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
