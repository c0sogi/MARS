import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_dataset
from library.model import RNAModel


# ==========================================
# LOSS FUNCTION
# ==========================================
def mcrmse_loss(pred, target):
    """
    Calculates the MCRMSE loss for training.
    pred: (Batch, Seq_Len, 5)
    target: (Batch, Seq_Len, 5)
    """
    # Calculate MSE over batch and sequence dimensions (dim 0 and 1)
    # Result shape: (5,)
    mse = torch.mean((pred - target) ** 2, dim=(0, 1))
    rmse = torch.sqrt(mse + 1e-8)  # Add epsilon for stability

    # Average RMSE across the 5 columns
    return torch.mean(rmse)


# ==========================================
# TRAINING
# ==========================================
def train_fn():
    # 1. Data Loading
    train_ds = get_dataset("train")
    val_ds = get_dataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Setup
    model = RNAModel().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 3. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            y = batch["y"].to(Config.DEVICE)  # Shape: (B, 68, 5)

            optimizer.zero_grad()

            # Forward pass
            # Output shape: (B, 107, 5)
            pred = model(x, adj)

            # Slice prediction to match target length (seq_scored=68)
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            loss = mcrmse_loss(pred_scored, y)
            loss.backward()

            # Gradient Clipping (Critical for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

            optimizer.step()
            running_loss += loss.item()

        # Update scheduler
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        val_mcrmse_all, val_mcrmse_scored = validate_fn(model, val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE (All): {val_mcrmse_all:.6f} | "
            f"Val MCRMSE (Scored): {val_mcrmse_scored:.6f}"
        )

        # Checkpointing & Early Stopping
        # We optimize based on the scored columns (Reactivity, Deg_Mg_pH10, Deg_Mg_50C)
        if val_mcrmse_scored < best_score:
            best_score = val_mcrmse_scored
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            # print(f"  New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                )
                break

    print(f"Training complete. Best Scored MCRMSE: {best_score:.6f}")


# ==========================================
# VALIDATION
# ==========================================
def validate_fn(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            y = batch["y"].to(Config.DEVICE)

            pred = model(x, adj)

            # Slice to scored length
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate to calculate global metric (avoids batch size bias)
    preds = np.concatenate(all_preds, axis=0)  # (N_samples, 68, 5)
    targets = np.concatenate(all_targets, axis=0)  # (N_samples, 68, 5)

    # Calculate MCRMSE
    # 1. MSE per column (averaging over samples and sequence length)
    mse = np.mean((preds - targets) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)

    # Global MCRMSE (all 5 columns)
    mcrmse_all = np.mean(rmse)

    # Scored MCRMSE (Columns 0, 1, 3: reactivity, deg_Mg_pH10, deg_Mg_50C)
    scored_indices = [0, 1, 3]
    mcrmse_scored = np.mean(rmse[scored_indices])

    return mcrmse_all, mcrmse_scored


# ==========================================
# INFERENCE
# ==========================================
def predict_fn(model, loader):
    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            ids.extend(batch["id"])

            # Predict for full sequence length (107)
            pred = model(x, adj)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N_test, 107, 5)
    return ids, preds


def generate_submission():
    print("Generating submission...")

    # 1. Load Test Data
    test_ds = get_dataset("test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Best Model
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file {Config.MODEL_PATH} not found. Run training first.")
        return

    model = RNAModel().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # 3. Predict
    ids, preds = predict_fn(model, test_loader)

    # 4. Format Submission
    # We need to flatten the predictions: one row per sequence position
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    id_seqpos_list = []
    flat_preds = []

    # preds shape: (N_samples, 107, 5)
    for i, sample_id in enumerate(ids):
        sample_pred = preds[i]  # (107, 5)
        for j in range(Config.SEQ_LEN):
            id_seqpos_list.append(f"{sample_id}_{j}")
            flat_preds.append(sample_pred[j])

    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    submission_df = pd.DataFrame(flat_preds, columns=cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # 5. Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
