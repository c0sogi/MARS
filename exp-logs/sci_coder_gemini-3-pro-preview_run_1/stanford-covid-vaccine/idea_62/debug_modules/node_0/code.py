import os
import torch
import numpy as np
import pandas as pd
import random
import shutil

# Import library components
from library.config import Config
from library.dataset import get_dataloaders
from library.model import RNA_Model
from library.loss import MaskedMSELoss, mcrmse
from library.engine import fit


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_demo():
    print("Initializing Demo...")

    # 1. Configure for Speed and Isolation
    # We modify Config attributes directly before they are used by other modules
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for speed (2 batches of 32)
    Config.EPOCHS = 2
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.create_dirs()

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n[Demo] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Data Structure
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")

    # Assertions for Input Shapes
    # Seq: (B, L)
    assert batch["seq"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Seq shape mismatch: {batch['seq'].shape}"
    # Dist: (B, L, Embed_Dist)
    assert batch["dist"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.EMBED_DIM_DIST,
    ), f"Dist shape mismatch: {batch['dist'].shape}"
    # Target: (B, 68, 3) - Targets are only defined for first 68 positions in training logic
    # Note: dataset.py loads full columns, but let's check what it returns.
    # Looking at dataset.py: targets = np.stack(..., axis=-1). It takes full length from DF.
    # However, the DF in metadata usually has length 68 arrays for targets.
    # Let's verify the actual shape returned.
    target_shape = batch["target"].shape
    print(f"Target shape: {target_shape}")
    assert target_shape[0] == Config.BATCH_SIZE
    assert target_shape[2] == Config.OUTPUT_DIM  # Should be 3

    print("Data Loading Verified.")

    # 3. Model Initialization and Forward Pass
    print("\n[Demo] Initializing Model...")
    model = RNA_Model().to(device)

    # Move batch to device
    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    dist = batch["dist"].to(device)
    targets = batch["target"].to(device)

    # Forward Pass
    preds = model(seq, loop, dist)
    print(f"Prediction shape: {preds.shape}")

    # Assert Output Shape: (B, L, Output_Dim)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.OUTPUT_DIM,
    ), "Model output shape mismatch"

    print("Model Forward Pass Verified.")

    # 4. Loss Calculation
    print("\n[Demo] Calculating Loss...")
    criterion = MaskedMSELoss()

    # Compute Loss
    loss = criterion(preds, targets)
    print(f"Loss value: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # Metric Check
    metric = mcrmse(preds, targets)
    print(f"MCRMSE score: {metric}")
    assert isinstance(metric, float), "MCRMSE should return a float"

    print("Loss Logic Verified.")

    # 5. Training Loop Execution
    print("\n[Demo] Starting Training Loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_score = fit(
        model, train_loader, val_loader, optimizer, scheduler, device, Config.EPOCHS
    )

    print(f"Training finished with Best MCRMSE: {best_score}")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_model.pth")
    ), "Model checkpoint not found"

    print("Training Loop Verified.")

    # 6. Inference and Submission
    print("\n[Demo] Generating Submission...")

    # Load Best Model
    model.load_state_dict(
        torch.load(
            os.path.join(Config.WORKING_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    preds_list = []
    ids_list = []

    # Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            # Forward
            out = model(seq, loop, dist)  # (B, 107, 3)
            out = out.cpu().numpy()

            preds_list.append(out)
            ids_list.extend(ids)

    # Concatenate Predictions
    preds_arr = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 3)

    # Reshape for Submission: We need one row per sequence position
    # The submission format expects: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Our model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)

    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_arr[i]  # (107, 3)
        for seqpos in range(Config.SEQ_LENGTH):
            # Construct row ID
            row_id = f"{sample_id}_{seqpos}"

            # Get predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unpredicted columns with 0.0 (as they are not scored but required in format)
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    # Create DataFrame
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=cols)

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Submission shape: {sub_df.shape}")

    # Validate Submission Shape
    # Expected rows: N_test_samples * 107
    # In DEBUG mode, N_test_samples is Config.DEBUG_SUBSET_SIZE (64)
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Submission Logic Verified.")
    print("\nAll Demo Steps Completed Successfully.")


if __name__ == "__main__":
    run_demo()
