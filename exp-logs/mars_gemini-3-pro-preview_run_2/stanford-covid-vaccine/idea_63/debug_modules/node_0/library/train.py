import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SCORED_LENGTH,
    SCORED_COLS,
    TARGET_COLS,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
    CACHE_DIR,
    SEED,
    FEEDBACK_CHANNELS,
    WORKING_DIR,
)
from library.data import get_dataloaders
from library.model import HS_GFDN
from library.loss import MaskedMCRMSE
from library.config import seed_everything


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training using the Iterative Refinement Loop.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # --- 1. Static Path (Compute Z once) ---
        # HS_GFDN.forward internally permutes inputs. We must do it manually here
        # to access the stem and backbone separately.
        # inputs: (N, L, 18) -> (N, 18, L)
        x_permuted = inputs.permute(0, 2, 1)
        x_stem = model.stem(x_permuted)
        z = model.backbone(x_stem)  # (N, Latent, L)

        # --- 2. Pass 1: Zero Feedback ---
        # Construct zero feedback embeddings
        N, _, L = z.shape
        e_fb_0 = torch.zeros((N, FEEDBACK_CHANNELS, L), device=device, dtype=z.dtype)

        y_hat_1 = model.head(z, e_fb_0, partner_indices)

        # --- 3. Pass 2: Recycled Feedback ---
        # Detach gradients from Pass 1 predictions to stop gradient flow through feedback generation
        # This stabilizes the RNN in the second pass.
        feedback = y_hat_1.detach()

        # Generate feedback embeddings from previous predictions
        # feedback_module expects (N, L, 5) and handles permutation internally
        e_fb_1 = model.feedback_module(feedback)

        # Interaction & Output using refined feedback
        y_hat_2 = model.head(z, e_fb_1, partner_indices)

        # --- 4. Loss Calculation ---
        # Weighted loss: Primary focus on final prediction, auxiliary supervision on first pass
        loss_2 = criterion(y_hat_2, targets)
        loss_1 = criterion(y_hat_1, targets)

        loss = loss_2 + 0.5 * loss_1

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Validates the model using Correct Global RMSE.
    Accumulates SSE and counts across the entire validation set.
    """
    model.eval()

    # Indices for scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [i for i, col in enumerate(TARGET_COLS) if col in SCORED_COLS]

    total_sse = torch.zeros(len(scored_indices), device=device)
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # --- 1. Static Path ---
            x_permuted = inputs.permute(0, 2, 1)
            x_stem = model.stem(x_permuted)
            z = model.backbone(x_stem)

            # --- 2. Pass 1 ---
            N, _, L = z.shape
            e_fb_0 = torch.zeros(
                (N, FEEDBACK_CHANNELS, L), device=device, dtype=z.dtype
            )
            y_hat_1 = model.head(z, e_fb_0, partner_indices)

            # --- 3. Pass 2 ---
            # Use Pass 1 output as feedback
            feedback = y_hat_1
            e_fb_1 = model.feedback_module(feedback)
            y_hat_2 = model.head(z, e_fb_1, partner_indices)

            # --- 4. Metric Accumulation ---
            # Select scored positions (0-67) and columns
            preds_masked = y_hat_2[:, :SCORED_LENGTH, :]
            targets_masked = targets[:, :SCORED_LENGTH, :]

            preds_selected = preds_masked[:, :, scored_indices]
            targets_selected = targets_masked[:, :, scored_indices]

            # Squared Error
            se = (preds_selected - targets_selected) ** 2

            # Sum over batch and sequence length, keep column dimension
            batch_sse = torch.sum(se, dim=(0, 1))

            total_sse += batch_sse
            total_count += inputs.size(0) * SCORED_LENGTH

    # Compute global RMSE per column
    mse_per_col = total_sse / total_count
    rmse_per_col = torch.sqrt(mse_per_col)

    # MCRMSE is mean of RMSEs across the scored columns
    mcrmse = torch.mean(rmse_per_col).item()

    return mcrmse


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["ids"]  # List of strings

            # --- 1. Static Path ---
            x_permuted = inputs.permute(0, 2, 1)
            x_stem = model.stem(x_permuted)
            z = model.backbone(x_stem)

            # --- 2. Pass 1 ---
            N, _, L = z.shape
            e_fb_0 = torch.zeros(
                (N, FEEDBACK_CHANNELS, L), device=device, dtype=z.dtype
            )
            y_hat_1 = model.head(z, e_fb_0, partner_indices)

            # --- 3. Pass 2 ---
            feedback = y_hat_1
            e_fb_1 = model.feedback_module(feedback)
            y_hat_2 = model.head(z, e_fb_1, partner_indices)

            # Store predictions (N, L, 5)
            all_preds.append(y_hat_2.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def generate_submission(preds, ids, submission_path):
    """
    Formats predictions into the competition CSV format.
    """
    # preds: (N_samples, Seq_Len, 5)
    # ids: list of N_samples strings

    data_rows = []
    cols = TARGET_COLS  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)
        length = sample_preds.shape[0]

        for pos in range(length):
            # Format: id_sequence_id_pos
            # sample_id is like "id_00b436dec"
            row_id = f"{sample_id}_{pos}"
            row_values = sample_preds[pos]

            # Create dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(cols):
                row_dict[col_name] = float(row_values[col_idx])

            data_rows.append(row_dict)

    df_sub = pd.DataFrame(data_rows)

    # Ensure column order matches sample submission
    final_cols = ["id_seqpos"] + cols
    df_sub = df_sub[final_cols]

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training():
    """
    Main execution function.
    """
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = HS_GFDN().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = MaskedMCRMSE()

    # Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step(val_mcrmse)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.20f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! ({val_mcrmse:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on test set...")
    preds, ids = inference(model, test_loader, device)

    print("Saving submission...")
    generate_submission(preds, ids, SUBMISSION_PATH)
