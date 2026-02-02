import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, MCRMSE, create_submission_file
from library.data import get_dataloaders
from library.model import DeepHierarchicalBiGRU
from library.layers import ConvStem, StabilizedGLUInteraction, RegressionHead
from library.train import Trainer


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up configuration...")
    seed_everything(42)

    # Override Config for a lightweight run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Reduce model size
    Config.HIDDEN_DIM = 64  # Reduced from 384
    Config.NUM_LAYERS = 2  # Reduced from 4

    # Reduce data size
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Verify Metric Logic (MCRMSE)
    print("\n[2] Verifying Metric (MCRMSE)...")
    # Scenario:
    # Seq Len = 107, Scored Len = 68
    # Targets = 5, Scored Cols indices = [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # We create a case where error is 1.0 for scored positions/cols and 100.0 for ignored ones.
    # The result should be exactly 1.0.

    batch_size = 2
    y_true = torch.zeros(batch_size, 107, 5)
    y_pred = torch.zeros(batch_size, 107, 5)

    # Set error = 1.0 for scored regions
    # Indices: 0, 1, 3 are scored. 2, 4 are ignored.
    scored_indices = [0, 1, 3]
    y_pred[:, :68, scored_indices] = 1.0

    # Set high error for unscored regions (should be ignored)
    y_pred[:, 68:, :] = 100.0
    y_pred[:, :68, [2, 4]] = 100.0

    score = MCRMSE(y_true, y_pred)
    print(f"   Calculated MCRMSE: {score:.4f}")

    # RMSE of 1.0 is 1.0. Mean of RMSEs (all 1.0) is 1.0.
    assert (
        abs(score - 1.0) < 1e-5
    ), f"Metric verification failed. Expected 1.0, got {score}"
    print("   Metric logic verified.")

    # 3. Verify Layers
    print("\n[3] Verifying Custom Layers...")

    # A. ConvStem
    stem = ConvStem(input_channels=14, kernel_size=3, filters=32)
    dummy_input = torch.randn(4, 107, 14)  # (B, L, C)
    stem_out = stem(dummy_input)
    print(f"   ConvStem Output Shape: {stem_out.shape}")
    assert stem_out.shape == (4, 107, 32), "ConvStem output shape mismatch"

    # B. StabilizedGLUInteraction
    # Needs hidden states and adjacency indices
    glu = StabilizedGLUInteraction(hidden_dim=32, dropout=0.0)
    dummy_hidden = torch.randn(4, 107, 32)
    # Create dummy adjacency: pair 0-5, 1-4, 2-3. Rest -1.
    dummy_adj = torch.full((4, 107), -1, dtype=torch.long)
    dummy_adj[:, 0] = 5
    dummy_adj[:, 5] = 0
    dummy_adj[:, 1] = 4
    dummy_adj[:, 4] = 1
    dummy_adj[:, 2] = 3
    dummy_adj[:, 3] = 2

    glu_out = glu(dummy_hidden, dummy_adj)
    print(f"   StabilizedGLUInteraction Output Shape: {glu_out.shape}")
    assert glu_out.shape == (4, 107, 32), "GLU Interaction output shape mismatch"

    # 4. Data Loading
    print("\n[4] Loading Data (Debug Mode)...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False,  # Force re-processing for demo
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Check one batch
    features, adjacency, targets = next(iter(train_loader))
    print(
        f"   Batch Shapes -> Features: {features.shape}, Adjacency: {adjacency.shape}, Targets: {targets.shape}"
    )

    assert (
        features.shape[1] == 107 and features.shape[2] == 14
    ), "Feature dimensions incorrect"
    assert adjacency.shape[1] == 107, "Adjacency dimensions incorrect"
    assert (
        targets.shape[1] == 107 and targets.shape[2] == 5
    ), "Target dimensions incorrect"
    print("   Data loading verified.")

    # 5. Model Initialization and Forward Pass
    print("\n[5] Initializing Model and Running Forward Pass...")
    device = "cpu"  # Use CPU for simple demo
    model = DeepHierarchicalBiGRU(
        input_channels=14,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=0.0,
    ).to(device)

    features = features.to(device)
    adjacency = adjacency.to(device)

    # Forward returns list of outputs (Deep Supervision)
    outputs = model(features, adjacency)

    print(f"   Number of outputs (Deep Supervision Heads): {len(outputs)}")
    assert (
        len(outputs) == Config.NUM_LAYERS
    ), "Number of outputs should match number of layers"

    final_output = outputs[-1]
    print(f"   Final Output Shape: {final_output.shape}")
    assert final_output.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), "Final output shape mismatch"
    print("   Model forward pass verified.")

    # 6. Training Loop Simulation
    print("\n[6] Simulating Training Loop...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, None, device, Config)

    # Run one epoch
    train_loss = trainer.train_epoch(train_loader)
    print(f"   Train Loss (1 Epoch): {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation
    val_score = trainer.validate(val_loader)
    print(f"   Validation MCRMSE: {val_score:.6f}")
    assert not np.isnan(val_score), "Validation score is NaN"

    # Save dummy checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("   Training simulation complete.")

    # 7. Submission Generation
    print("\n[7] Generating Submission File...")
    # Create dummy predictions for test set
    num_test_samples = len(test_ids)
    test_preds = np.random.rand(num_test_samples, 107, 5).astype(np.float32)

    create_submission_file(test_ids, test_preds, Config.SUBMISSION_PATH)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission file created. Shape: {df_sub.shape}")

        # Expected rows: num_test_samples * 107
        expected_rows = num_test_samples * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

        # Check columns
        expected_cols = ["id_seqpos"] + Config.get_target_columns()
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
