import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.data import get_dataset, RNADataset
from library.modules import AHCHDN
from library.utils import seed_everything, calculate_global_rmse


def mcrmse_loss(pred, target):
    """
    Calculates Mean Columnwise Root Mean Squared Error for training.
    Averages the RMSE of each column.

    Args:
        pred (torch.Tensor): Predictions of shape (N, L, 5)
        target (torch.Tensor): Targets of shape (N, L, 5)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Squared Error
    mse = (pred - target) ** 2
    # Mean over Batch (0) and Sequence (1) -> (5,)
    mse_per_col = torch.mean(mse, dim=(0, 1))
    # RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)
    # Mean over columns
    return torch.mean(rmse_per_col)


def train_model():
    """
    Executes the training pipeline for AHC-HDN.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Data
    # Uses caching mechanism from library.data
    train_data = get_dataset("train", load_cached_data=True)
    val_data = get_dataset("val", load_cached_data=True)

    train_ds = RNADataset(train_data, "train")
    val_ds = RNADataset(val_data, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. Initialize Model, Optimizer, Scheduler
    model = AHCHDN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE // 2
    )

    best_score = float("inf")
    best_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    # 3. Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for features, pair_map, targets in train_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # --- Iterative Refinement Loop ---

            # Pass 1: Initial prediction with zero feedback
            y1 = model(features, pair_map, y_prev=None)

            # Pass 2: Refinement using detached predictions from Pass 1
            # Note: y1.detach() prevents gradients flowing through the feedback generation of Pass 1
            y2 = model(features, pair_map, y_prev=y1.detach())

            # --- Anchored Loss ---
            # Calculated over the full sequence length (0-107).
            # The targets for tail positions (68-107) are 0.0 (Neutral Baseline).
            loss = mcrmse_loss(y2, targets) + 0.5 * mcrmse_loss(y1, targets)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for features, pair_map, targets in val_loader:
                features = features.to(device)
                pair_map = pair_map.to(device)

                # Inference Strategy: 2 Passes
                y1 = model(features, pair_map, y_prev=None)
                y2 = model(features, pair_map, y_prev=y1)

                val_preds.append(y2.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        # Metric: Correct Global RMSE
        # Calculated ONLY on scored positions (0-68) and scored columns
        val_score = calculate_global_rmse(
            val_preds,
            val_targets,
            scored_length=Config.SCORED_LENGTH,
            scored_cols_indices=Config.SCORED_COLS_INDICES,
        )

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Val Score: {best_score:.6f}")


def run_inference():
    """
    Executes inference on the test set and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Inference...")

    model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Model
    model = AHCHDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_data = get_dataset("test", load_cached_data=True)
    test_ds = RNADataset(test_data, "test")
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    preds_list = []
    with torch.no_grad():
        for features, pair_map in test_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)

            # Inference Strategy: 2 Passes
            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)

            preds_list.append(y2.cpu().numpy())

    # Concatenate all predictions: (N_test, 107, 5)
    preds = np.concatenate(preds_list, axis=0)
    ids = test_data["ids"]

    # Flatten predictions for Submission Format
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []
    for i, sample_id in enumerate(ids):
        for j in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{j}"
            row_vals = preds[i, j, :]
            row = [row_id] + row_vals.tolist()
            submission_data.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_data, columns=columns)

    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
