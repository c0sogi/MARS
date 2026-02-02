import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, MCRMSELoss, GlobalMetrics
from library.model import GC_SSN
from library.data import get_dataloaders, RNADataset, load_dataset


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configure for Fast Demonstration
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 64  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_TRAIN = os.path.join(Config.WORKING_DIR, "debug_cache_train.npz")
    Config.CACHE_VAL = os.path.join(Config.WORKING_DIR, "debug_cache_val.npz")
    Config.CACHE_TEST = os.path.join(Config.WORKING_DIR, "debug_cache_test.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for speed and reproducibility.")

    # 2. Data Loading Demonstration
    print("\n[2] Demonstrating Data Loading...")
    # We force reload to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debug script to avoid multiprocessing overhead
    )

    # Fetch one batch to verify shapes
    try:
        x_batch, bpp_batch, y_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"Batch shapes captured:")
    print(f"  Input Features (x): {x_batch.shape}")
    print(f"  BPP Indices (bpp): {bpp_batch.shape}")
    print(f"  Targets (y): {y_batch.shape}")

    # Assertions
    # x: (Batch, 18, 107)
    assert x_batch.shape == (
        Config.BATCH_SIZE,
        Config.INPUT_DIM,
        Config.SEQ_LEN,
    ), f"Expected x shape {(Config.BATCH_SIZE, Config.INPUT_DIM, Config.SEQ_LEN)}, got {x_batch.shape}"
    # bpp: (Batch, 107)
    assert bpp_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Expected bpp shape {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {bpp_batch.shape}"
    # y: (Batch, 107, 5)
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        len(Config.TARGET_COLS),
    ), f"Expected y shape {(Config.BATCH_SIZE, Config.SEQ_LEN, len(Config.TARGET_COLS))}, got {y_batch.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization & Forward Pass
    print("\n[3] Demonstrating Model Initialization & Forward Pass...")
    device = Config.DEVICE
    model = GC_SSN().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)
    bpp_batch = bpp_batch.to(device)
    y_batch = y_batch.to(device)

    # Forward pass
    # Model returns (y_final, y_aux)
    y_pred_final, y_pred_aux = model(x_batch, bpp_batch)

    print(f"Model Output Shape: {y_pred_final.shape}")

    # Assertions
    assert (
        y_pred_final.shape == y_batch.shape
    ), f"Output shape mismatch. Expected {y_batch.shape}, got {y_pred_final.shape}"
    assert (
        y_pred_aux.shape == y_batch.shape
    ), f"Aux output shape mismatch. Expected {y_batch.shape}, got {y_pred_aux.shape}"

    print("Model forward pass verified.")

    # 4. Loss Function Verification
    print("\n[4] Demonstrating Loss Calculation (MCRMSE)...")
    criterion = MCRMSELoss()

    loss = criterion(y_pred_final, y_batch)
    print(f"Calculated MCRMSE Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    print("Loss function verified.")

    # 5. Training Loop Simulation
    print("\n[5] Simulating Training Loop (5 Steps)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    model.train()

    initial_loss = loss.item()

    for step in range(5):
        optimizer.zero_grad()

        # Forward
        y_final, y_aux = model(x_batch, bpp_batch)

        # Loss (Combined final and aux)
        step_loss = criterion(y_final, y_batch) + 0.5 * criterion(y_aux, y_batch)

        # Backward
        step_loss.backward()

        # Update
        optimizer.step()

        print(f"  Step {step+1}: Loss = {step_loss.item():.6f}")

    print("Training loop simulation complete.")

    # 6. Global Metrics Verification
    print("\n[6] Demonstrating Global Metrics Calculation...")
    model.eval()
    global_metrics = GlobalMetrics()

    with torch.no_grad():
        # Simulate validation over a few batches
        for i, (vx, vbpp, vy) in enumerate(val_loader):
            if i >= 2:
                break  # Just 2 batches
            vx, vbpp, vy = vx.to(device), vbpp.to(device), vy.to(device)
            vy_pred, _ = model(vx, vbpp)
            global_metrics.update(vy_pred, vy)

    val_score = global_metrics.compute()
    print(f"Validation MCRMSE Score: {val_score:.6f}")
    assert isinstance(val_score, float), "Score should be a float"

    # 7. Inference & Submission Generation
    print("\n[7] Generating Submission File...")

    # Save a dummy best model for the prediction function to find, if we were using the library function directly.
    # However, we will manually run inference here to show how to use the components.

    submission_rows = []

    with torch.no_grad():
        for i, (tx, tbpp) in enumerate(test_loader):
            if i >= 2:
                break  # Limit to 2 batches for speed
            tx, tbpp = tx.to(device), tbpp.to(device)

            # Get predictions
            ty_pred, _ = model(tx, tbpp)  # (B, 107, 5)
            ty_pred = ty_pred.cpu().numpy()

            # We need IDs to map predictions.
            # In a real run, we'd iterate the dataset which has IDs, or index into the loaded data dict.
            # Here we will just generate dummy IDs based on batch index for demonstration of formatting.
            batch_size_curr = ty_pred.shape[0]
            start_idx = i * Config.BATCH_SIZE

            # Retrieve actual IDs from the dataset if possible, or mock them
            # The test_loader dataset is RNADataset. To get IDs we need the raw data dict or similar.
            # Let's access the underlying data from the loader's dataset
            dataset_ids = (
                test_loader.dataset.features
            )  # This is the features array, not IDs.
            # The IDs are stored in the `data` dict passed to RNADataset, but RNADataset doesn't store IDs as a property.
            # We will mock IDs for this demo to show the formatting logic.

            for b in range(batch_size_curr):
                sample_id = f"id_demo_{start_idx + b}"
                for pos in range(Config.SEQ_LEN):
                    row_id = f"{sample_id}_{pos}"
                    preds = ty_pred[b, pos]

                    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    # Model Output Order (Config.TARGET_COLS):
                    # [reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C]
                    # Indices: 0, 1, 2, 3, 4

                    # Submission Format Required:
                    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

                    vals = [
                        preds[0],  # reactivity
                        preds[1],  # deg_Mg_pH10
                        preds[3],  # deg_pH10 (Index 3 in model output)
                        preds[2],  # deg_Mg_50C (Index 2 in model output)
                        preds[4],  # deg_50C
                    ]
                    submission_rows.append([row_id] + vals)

    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=cols)

    print(f"Generated {len(sub_df)} rows of predictions.")
    print("Sample rows:")
    print(sub_df.head())

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
