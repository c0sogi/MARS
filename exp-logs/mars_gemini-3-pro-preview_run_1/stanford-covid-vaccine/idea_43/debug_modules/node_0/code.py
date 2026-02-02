import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data import get_loaders
from library.model import RNA_ResBiLSTM
from library.engine import run_training
from library.utils import mcrmse_metric


def main():
    print("--- RNA Degradation Prediction Demo ---")

    # ---------------------------------------------------------
    # 1. Configuration Override
    # ---------------------------------------------------------
    # Modify Config globally to optimize for a fast demo run
    print("[1/6] Configuring environment...")

    # Paths
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Model Hyperparameters (Reduced for speed)
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 2
    Config.EMBED_DIM_SEQ = 32
    Config.EMBED_DIM_LOOP = 16
    Config.EMBED_DIM_DIST = 16
    # Total input dim must match sum of embeddings
    Config.TOTAL_INPUT_DIM = (
        Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST
    )

    # Training Hyperparameters
    Config.EPOCHS = 1  # Single epoch for demo
    Config.BATCH_SIZE = 16  # Small batch size

    # Set Seed
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")
    print(f"      Working Dir: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("[2/6] Loading and verifying data...")

    # Force reload to demonstrate processing logic
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify Train Batch
    batch = next(iter(train_loader))
    seq, loop, dist, targets = (
        batch["seq"],
        batch["loop"],
        batch["dist"],
        batch["targets"],
    )

    # Assert Shapes
    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Seq shape mismatch: {seq.shape}"
    assert loop.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Loop shape mismatch: {loop.shape}"
    assert dist.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Dist shape mismatch: {dist.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SCORED_LEN,
        3,
    ), f"Targets shape mismatch: {targets.shape}"

    print("      Data shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # ---------------------------------------------------------
    print("[3/6] Initializing model...")

    model = RNA_ResBiLSTM(Config).to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Move batch to device
        b_seq = seq.to(device)
        b_loop = loop.to(device)
        b_dist = dist.to(device)

        preds = model(b_seq, b_loop, b_dist)

    # Model outputs predictions for full sequence length (107), not just scored length (68)
    expected_out_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, 3)
    assert (
        preds.shape == expected_out_shape
    ), f"Output shape mismatch. Expected {expected_out_shape}, got {preds.shape}"

    print("      Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Metric Verification
    # ---------------------------------------------------------
    print("[4/6] Verifying metric logic...")

    # Test MCRMSE with identical tensors (Should be 0.0)
    dummy_t = torch.rand(10, 68, 3)
    score_perfect = mcrmse_metric(dummy_t, dummy_t)
    assert score_perfect < 1e-6, f"Metric failed on perfect match: {score_perfect}"

    # Test MCRMSE with known offset (Offset 1.0 -> MSE 1.0 -> RMSE 1.0 -> MCRMSE 1.0)
    dummy_p = dummy_t + 1.0
    score_offset = mcrmse_metric(dummy_p, dummy_t)
    assert abs(score_offset - 1.0) < 1e-5, f"Metric failed on offset: {score_offset}"

    print("      Metric logic verified.")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("[5/6] Running training loop...")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Train
    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=save_path,
    )

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("[6/6] Generating submission...")

    model.eval()
    all_preds = []

    # Run Inference on Test Loader
    with torch.no_grad():
        for batch in test_loader:
            t_seq = batch["seq"].to(device)
            t_loop = batch["loop"].to(device)
            t_dist = batch["dist"].to(device)

            # (B, 107, 3)
            batch_preds = model(t_seq, t_loop, t_dist)
            all_preds.append(batch_preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Get Test IDs from the dataset
    test_ids = test_loader.dataset.ids
    assert len(all_preds) == len(test_ids), "Mismatch between predictions and test IDs"

    # Format Submission
    submission_data = []

    for i, sample_id in enumerate(test_ids):
        sample_pred = all_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Predicted columns: reactivity, deg_Mg_pH10, deg_Mg_50C
            reactivity = sample_pred[seqpos, 0]
            deg_Mg_pH10 = sample_pred[seqpos, 1]
            deg_Mg_50C = sample_pred[seqpos, 2]

            # Non-scored/Non-predicted columns filled with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    submission_df = pd.DataFrame(
        submission_data, columns=["id_seqpos"] + Config.ALL_SUBMISSION_COLS
    )

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"      Submission saved to {Config.SUBMISSION_FILE}")
    print("      Rows generated:", len(submission_df))
    print("Done.")


if __name__ == "__main__":
    main()
