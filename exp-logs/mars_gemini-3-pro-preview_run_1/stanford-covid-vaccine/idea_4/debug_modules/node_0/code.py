import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.dataset import get_dataloaders, RNADataset
from library.model import HybridCNNBiGRU
from library.loss import MaskedMSELoss
from library.train import Trainer, generate_submission


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Configure for Fast Demonstration
    # We modify the Config singleton to use a temporary directory and minimal parameters
    print("[1] Configuring environment for fast execution...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set debug mode to process only a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small number of samples
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")

    # 2. Data Loading and Verification
    print("\n[2] Loading and Verifying Data...")
    # Force processing from parquet (load_cached_data=False) to test preprocessing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch from training
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {
        "id",
        "sequence",
        "structure",
        "predicted_loop_type",
        "targets",
        "mask",
    }
    assert (
        set(batch.keys()) == expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Sequence: (Batch, 107)
    assert batch["sequence"].shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    # Targets: (Batch, 107, 5)
    assert batch["targets"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    )
    # Mask: (Batch, 107)
    assert batch["mask"].shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)

    # Verify Mask Logic
    # The mask should be 1 for the first SEQ_SCORED (68) positions and 0 afterwards
    mask_np = batch["mask"].numpy()
    assert np.all(
        mask_np[:, : Config.SEQ_SCORED] == 1.0
    ), "Mask should be 1.0 for scored positions"
    assert np.all(
        mask_np[:, Config.SEQ_SCORED :] == 0.0
    ), "Mask should be 0.0 for unscored positions"

    print("    Data shapes and mask logic verified.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Initializing Model and Verifying Forward Pass...")
    model = HybridCNNBiGRU().to(device)

    # Move batch to device
    seq = batch["sequence"].to(device)
    struct = batch["structure"].to(device)
    loop = batch["predicted_loop_type"].to(device)

    # Forward pass
    outputs = model(seq, struct, loop)

    # Verify Output Shape: (Batch, 107, 5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)}, got {outputs.shape}"

    print("    Model forward pass successful. Output shape verified.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Custom MaskedMSELoss...")
    criterion = MaskedMSELoss()

    # Test with the actual batch
    loss = criterion(outputs, batch["targets"].to(device), batch["mask"].to(device))
    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Test with Synthetic Data to ensure masking works
    # Case 1: Perfect prediction on masked area, huge error on unmasked area -> Loss should be 0
    synth_pred = torch.tensor([[[0.0], [100.0]]], device=device)  # Shape (1, 2, 1)
    synth_targ = torch.tensor([[[0.0], [0.0]]], device=device)  # Shape (1, 2, 1)
    synth_mask = torch.tensor(
        [[1.0, 0.0]], device=device
    )  # Shape (1, 2) - Only first pos valid

    synth_loss = criterion(synth_pred, synth_targ, synth_mask)
    assert torch.isclose(
        synth_loss, torch.tensor(0.0, device=device)
    ), f"Masking failed. Expected 0.0 loss, got {synth_loss.item()}"

    print("    MaskedMSELoss logic verified.")

    # 5. Metric Verification
    print("\n[5] Verifying MCRMSE Metric...")
    # Synthetic metric test
    # 2 samples, 2 targets.
    # Sample 1: Error 1.0 on both targets -> MSE=1, RMSE=1
    # Sample 2: Error 3.0 on both targets -> MSE=9, RMSE=3
    # Column 1 RMSE: sqrt((1+9)/2) = sqrt(5) = 2.236
    # Column 2 RMSE: sqrt((1+9)/2) = sqrt(5) = 2.236
    # MCRMSE: (2.236 + 2.236) / 2 = 2.236

    y_true = np.array([[0, 0], [0, 0]])
    y_pred = np.array([[1, 1], [3, 3]])

    metric_val = mcrmse_metric(y_true, y_pred)
    expected_val = np.sqrt(5)
    assert np.isclose(
        metric_val, expected_val, atol=1e-4
    ), f"Metric calculation failed. Expected {expected_val}, got {metric_val}"

    print("    MCRMSE metric calculation verified.")

    # 6. Training Loop Demonstration
    print("\n[6] Running Training Loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    trainer.fit(epochs=Config.EPOCHS, patience=2)

    # Check if model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Best model checkpoint was not saved."
    print("    Training loop completed and model saved.")

    # 7. Inference and Submission
    print("\n[7] Running Inference and Generating Submission...")
    # Load the best model weights
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Predict
    test_ids, test_preds = trainer.predict(test_loader)

    # Verify Prediction Shape
    # Number of test samples in debug mode is Config.DEBUG_SUBSET_SIZE (or less if file is smaller)
    # But wait, test.json has 240 lines. In Debug mode, we slice it.
    n_test_samples = len(test_ids)
    assert test_preds.shape == (
        n_test_samples,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Prediction shape mismatch."

    # Generate CSV
    generate_submission(test_ids, test_preds, Config.SUBMISSION_PATH)

    # Verify CSV content
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: n_test_samples * 107
    expected_rows = n_test_samples * Config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("    Submission generated and verified successfully.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
