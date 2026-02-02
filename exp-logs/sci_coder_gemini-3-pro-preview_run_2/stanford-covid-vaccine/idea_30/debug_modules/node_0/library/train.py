import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything
from library.data import get_loader
from library.model import SR_DCN

# ==========================================
# Helper Functions
# ==========================================


def masked_mcrmse(preds, targets, mask, scored_indices):
    """
    Calculates MCRMSE only on valid masked positions and scored columns.

    Args:
        preds (torch.Tensor): (B, L, 5)
        targets (torch.Tensor): (B, L, 5)
        mask (torch.Tensor): (B, L)
        scored_indices (list): List of column indices to score.

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Flatten tensors: (B*L, C)
    b, l, c = preds.shape
    preds_flat = preds.view(-1, c)
    targets_flat = targets.view(-1, c)
    mask_flat = mask.view(-1)

    # Filter by mask
    valid_indices = mask_flat > 0.5
    if valid_indices.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    preds_valid = preds_flat[valid_indices]
    targets_valid = targets_flat[valid_indices]

    # Select scored columns
    preds_scored = preds_valid[:, scored_indices]
    targets_scored = targets_valid[:, scored_indices]

    # MSE per column
    mse = torch.mean((preds_scored - targets_scored) ** 2, dim=0)

    # RMSE per column
    rmse = torch.sqrt(mse + 1e-8)

    # Mean of RMSEs
    return torch.mean(rmse)


# ==========================================
# Training & Validation
# ==========================================


def train_epoch(model, loader, optimizer, device, scored_indices):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y, p_idx, mask, ids) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        p_idx = p_idx.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # --- Pass 1: Cold Start ---
        # Initialize recycling channels to zero
        b, l, _ = x.shape
        recycling_zero = torch.zeros((b, l, 5), device=device)

        pred1 = model(x, recycling_zero, p_idx)

        # --- Pass 2: Refinement ---
        # Detach Pass 1 output to stop gradients flowing back through the recycling loop
        # This stabilizes the training (Stabilized Recycling)
        recycling_input = pred1.detach()

        pred2 = model(x, recycling_input, p_idx)

        # --- Loss Calculation ---
        # Calculate loss on scored columns only
        loss_primary = masked_mcrmse(pred2, y, mask, scored_indices)
        loss_aux = masked_mcrmse(pred1, y, mask, scored_indices)

        # Combined loss
        loss = loss_primary + Config.AUX_LOSS_WEIGHT * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, scored_indices):
    model.eval()

    # Accumulators for Global MCRMSE
    # We need to sum squared errors per column across the whole dataset
    total_se = torch.zeros(len(scored_indices), device=device)
    total_count = 0

    with torch.no_grad():
        for x, y, p_idx, mask, ids in loader:
            x = x.to(device)
            y = y.to(device)
            p_idx = p_idx.to(device)
            mask = mask.to(device)

            # --- Pass 1: Cold Start ---
            b, l, _ = x.shape
            recycling_zero = torch.zeros((b, l, 5), device=device)
            pred1 = model(x, recycling_zero, p_idx)

            # --- Pass 2: Refinement ---
            recycling_input = (
                pred1  # No detach needed in eval, but logically consistent
            )
            pred2 = model(x, recycling_input, p_idx)

            # --- Accumulate Errors ---
            # Flatten
            preds_flat = pred2.view(-1, 5)
            targets_flat = y.view(-1, 5)
            mask_flat = mask.view(-1)

            valid_mask = mask_flat > 0.5
            if valid_mask.sum() == 0:
                continue

            preds_valid = preds_flat[valid_mask]
            targets_valid = targets_flat[valid_mask]

            # Select scored columns
            preds_scored = preds_valid[:, scored_indices]
            targets_scored = targets_valid[:, scored_indices]

            # Sum Squared Errors
            se = torch.sum((preds_scored - targets_scored) ** 2, dim=0)
            total_se += se
            total_count += preds_scored.shape[0]

    # Compute Global RMSE
    if total_count == 0:
        return 0.0

    mse = total_se / total_count
    rmse = torch.sqrt(mse + 1e-8)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


def train_model():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # The target tensor has 5 columns: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    scored_indices = [0, 1, 3]

    # Load Data
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)

    # Initialize Model
    model = SR_DCN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, scored_indices)
        val_loss = validate(model, val_loader, device, scored_indices)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_loss:.10f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print("  Saved Best Model")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_val_loss:.10f}")


# ==========================================
# Inference & Submission
# ==========================================


def predict_and_submit():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_loader = get_loader("test", shuffle=False)

    # Load Model
    model = SR_DCN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using initialized weights.")

    model.eval()

    results = []

    # Target column names in order of model output (5 channels)
    target_cols = Config.TARGET_COLS

    print("Generating predictions...")
    with torch.no_grad():
        for x, y, p_idx, mask, ids in test_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)

            # --- Pass 1: Cold Start ---
            b, l, _ = x.shape
            recycling_zero = torch.zeros((b, l, 5), device=device)
            pred1 = model(x, recycling_zero, p_idx)

            # --- Pass 2: Refinement ---
            recycling_input = pred1
            pred2 = model(x, recycling_input, p_idx)

            # Move to CPU
            preds_np = pred2.cpu().numpy()  # (B, L, 5)

            # Process batch
            for i, sample_id in enumerate(ids):
                # We need to output predictions for all sequence positions (0 to 106)
                # The model outputs (107, 5)
                sample_preds = preds_np[i]  # (107, 5)

                for seqpos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"
                    vals = sample_preds[seqpos]

                    # Create row dictionary
                    row_dict = {"id_seqpos": row_id}
                    for k, col_name in enumerate(target_cols):
                        row_dict[col_name] = float(vals[k])

                    results.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Reorder columns to match submission format
    # id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    train_model()
    predict_and_submit()
