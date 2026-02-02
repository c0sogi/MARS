import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.data_utils import process_data
from library.dataset import RNADataset, get_dataloader
from library.model_components import HSDARNModel
from library.loss_metric import MCRMSELoss, GlobalMCRMSE
from library.train_eval import Trainer


def main():
    print("=== Starting Demonstration of HS-DARN Library ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # -------------------------------------------------------------------------
    print("1. Configuring environment for demo...")

    # Set a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Modify Config static variables
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.DEBUG = True  # Enable debug mode to load subsets
    Config.DEBUG_SUBSET_SIZE = 20  # Small subset size
    Config.BATCH_SIZE = 4  # Small batch size
    Config.CACHE_KEY = "demo_cache"  # Unique cache key

    # Set seed for reproducibility
    Config.set_seed(42)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print("   Configuration complete.\n")

    # -------------------------------------------------------------------------
    # 2. Data Processing Verification (library.data_utils)
    # -------------------------------------------------------------------------
    print("2. Verifying Data Processing (library.data_utils)...")

    # Test processing training data
    inputs, partner_map, targets, ids = process_data(
        data_type="train",
        load_cached_data=False,  # Force processing from source
        debug=True,
    )

    # Assertions for shapes
    # Inputs: (N, Seq_Len=107, Channels=18)
    assert inputs.ndim == 3, f"Inputs should be 3D, got {inputs.ndim}"
    assert inputs.shape[1] == 107, f"Seq len should be 107, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == 18
    ), f"Feature channels should be 18, got {inputs.shape[2]}"

    # Partner Map: (N, 107)
    assert partner_map.shape == (inputs.shape[0], 107), "Partner map shape mismatch"

    # Targets: (N, Scored_Len=68, Channels=5)
    assert targets is not None, "Targets should not be None for train data"
    assert (
        targets.shape[1] == 68
    ), f"Target seq len should be 68, got {targets.shape[1]}"
    assert targets.shape[2] == 5, f"Target channels should be 5, got {targets.shape[2]}"

    print(f"   Processed {len(ids)} training samples successfully.")
    print("   Data Processing verification passed.\n")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Verification (library.dataset)
    # -------------------------------------------------------------------------
    print("3. Verifying Dataset and DataLoader (library.dataset)...")

    # Instantiate Dataset manually
    ds = RNADataset(inputs, partner_map, targets, ids)
    sample = ds[0]

    # Check item keys and types
    assert "inputs" in sample
    assert "partner_map" in sample
    assert "targets" in sample
    assert isinstance(sample["inputs"], torch.Tensor)
    assert sample["inputs"].shape == (107, 18)

    # Test DataLoader
    loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, debug=True)
    batch = next(iter(loader))

    assert batch["inputs"].shape[0] == Config.BATCH_SIZE
    assert batch["inputs"].shape[1] == 107
    assert batch["inputs"].shape[2] == 18

    print("   Dataset and DataLoader verification passed.\n")

    # -------------------------------------------------------------------------
    # 4. Model Component Verification (library.model_components)
    # -------------------------------------------------------------------------
    print("4. Verifying Model Architecture (library.model_components)...")

    model = HSDARNModel()
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy inputs on correct device
    dummy_inputs = torch.randn(2, 107, 18).to(Config.DEVICE)
    dummy_pmap = torch.zeros(2, 107, dtype=torch.long).to(Config.DEVICE)

    with torch.no_grad():
        # Forward pass returns tuple (y_hat_1, y_hat_2)
        y1, y2 = model(dummy_inputs, dummy_pmap)

    # Verify outputs
    # Output shape should be (Batch, Seq_Len=107, Output_Dim=5)
    expected_shape = (2, 107, 5)
    assert (
        y1.shape == expected_shape
    ), f"y1 shape mismatch: {y1.shape} != {expected_shape}"
    assert (
        y2.shape == expected_shape
    ), f"y2 shape mismatch: {y2.shape} != {expected_shape}"

    print("   Model forward pass verification passed.\n")

    # -------------------------------------------------------------------------
    # 5. Loss and Metric Verification (library.loss_metric)
    # -------------------------------------------------------------------------
    print("5. Verifying Loss and Metric (library.loss_metric)...")

    criterion = MCRMSELoss().to(Config.DEVICE)

    # Create dummy predictions (Batch, 107, 5) and targets (Batch, 68, 5)
    # Note: Loss function handles slicing internally
    pred_t = torch.randn(4, 107, 5).to(Config.DEVICE)
    target_t = torch.randn(4, 68, 5).to(Config.DEVICE)

    loss = criterion(pred_t, target_t)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Test Global Metric Accumulator
    metric = GlobalMCRMSE()
    metric.update(pred_t, target_t)
    score = metric.compute()
    assert isinstance(score, float), "Global metric should return a float"

    print(f"   Loss calculation: {loss.item():.4f}")
    print(f"   Global Metric calculation: {score:.4f}")
    print("   Loss and Metric verification passed.\n")

    # -------------------------------------------------------------------------
    # 6. Full Pipeline Execution (library.train_eval)
    # -------------------------------------------------------------------------
    print("6. Executing Full Training Pipeline (library.train_eval)...")

    # Instantiate Trainer
    # This will initialize the model, optimizer, etc.
    trainer = Trainer()

    # Run Training (fit)
    # With EPOCHS=1 and DEBUG=True, this should be very fast
    print("   Running trainer.fit()...")
    trainer.fit()

    # Check if model checkpoint was created
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("   Model checkpoint confirmed.")

    # Run Inference (predict)
    print("   Running trainer.predict()...")
    trainer.predict()

    # -------------------------------------------------------------------------
    # 7. Submission Validation
    # -------------------------------------------------------------------------
    print("7. Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission shape: {df_sub.shape}")
    print(f"   Submission columns: {df_sub.columns.tolist()}")

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Expected rows:
    # In debug mode, 'test' loader loads DEBUG_SUBSET_SIZE samples.
    # Each sample has 107 positions.
    # Total rows = DEBUG_SUBSET_SIZE * 107
    expected_rows = Config.DEBUG_SUBSET_SIZE * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check for NaN
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("   Submission file validation passed.\n")

    print("=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
