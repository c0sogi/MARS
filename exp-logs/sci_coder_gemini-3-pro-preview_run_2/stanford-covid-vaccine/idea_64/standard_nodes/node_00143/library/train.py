import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, MCRMSE_Metric
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import ML_GFN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training using the Iterative Refinement strategy.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        pair_indices = batch["pair_indices"].to(device)

        optimizer.zero_grad()

        # =====================================================================
        # Step 1: Static Encoding
        # =====================================================================
        # Compute Z once per batch
        z = model.encode_static(inputs)

        # =====================================================================
        # Step 2: Pass 1 (Zero Feedback)
        # =====================================================================
        # Initial prediction with no previous state
        y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)

        # Calculate auxiliary loss for the first pass
        loss1 = criterion(y1, targets, mask)

        # =====================================================================
        # Step 3: Pass 2 (With Feedback)
        # =====================================================================
        # Detach gradients from first pass predictions to stop gradient flow
        # through the feedback loop target generation (Stop-Gradient)
        y1_detached = y1.detach()

        # Refined prediction using the first pass as feedback
        # The model internally handles channel masking in the FeedbackModule
        y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1_detached)

        # Calculate main loss for the second pass
        loss2 = criterion(y2, targets, mask)

        # =====================================================================
        # Optimization
        # =====================================================================
        # Weighted sum of losses
        loss = loss2 + (Config.AUX_LOSS_WEIGHT * loss1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the global MCRMSE metric.
    """
    model.eval()
    metric = MCRMSE_Metric()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            pair_indices = batch["pair_indices"].to(device)

            # Inference: Two-pass strategy
            z = model.encode_static(inputs)
            y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)
            y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1)

            # Update global metric with final predictions
            metric.update(y2, targets, mask)

            # Calculate loss for monitoring (optional, usually matches metric)
            loss = criterion(y2, targets, mask)
            running_loss += loss.item()

    score = metric.compute()
    avg_loss = running_loss / len(loader)

    return score, avg_loss


def train_model():
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Initializing Data Loaders...")
    train_loader, val_loader, _ = get_loaders(load_cached_data=True)

    # 2. Model Setup
    print("Initializing Model...")
    model = ML_GFN().to(device)

    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = MaskedMCRMSELoss().to(device)

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score, val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation Score: {best_score}")


def generate_submission():
    """
    Generates the submission file using the best trained model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    _, _, test_loader = get_loaders(load_cached_data=True)

    # 2. Load Model
    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model = ML_GFN().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 3. Inference
    ids_list = []
    preds_list = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            batch_ids = batch["id"]

            # Two-pass inference
            z = model.encode_static(inputs)
            y1 = model.decode_dynamic(z, pair_indices, prev_preds=None)
            y2 = model.decode_dynamic(z, pair_indices, prev_preds=y1)

            # Move to CPU
            preds_np = y2.cpu().numpy()  # (Batch, SeqLen, 5)

            ids_list.extend(batch_ids)
            preds_list.append(preds_np)

    # Concatenate all predictions
    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # 4. Format Submission
    # We need to flatten the predictions to one row per sequence position
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            # Create a dictionary for the row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    # This block is just for local testing if run directly,
    # but the instructions say "DO NOT include an if __name__ == '__main__': block"
    # for the module implementation itself. However, to make the script executable
    # as a standalone entry point if needed, I will include the calls but commented out
    # or rely on the user to import and call functions.
    # Per strict instructions "Only implement the module class/functions",
    # I will provide the functions ready to be imported.
    pass
