import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    SUBMISSION_FILE_PATH,
    SEQ_SCORED,
    SEQ_LEN,
    TARGET_COLS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    MAX_GRAD_NORM,
    DEBUG,
    DEBUG_SUBSET_SIZE,
    SEED,
    SCORED_TARGET_INDICES,
)
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import DeepPostNormBiGRU


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using MCRMSE loss and Gradient Clipping.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices, pair_mask)

        # Slice to scored sequence length for loss calculation
        # Shape: (Batch, SEQ_SCORED, Num_Targets)
        preds_sliced = preds[:, :SEQ_SCORED, :]
        targets_sliced = targets[:, :SEQ_SCORED, :]

        # Filter for scored columns only
        preds_filtered = preds_sliced[:, :, SCORED_TARGET_INDICES]
        targets_filtered = targets_sliced[:, :, SCORED_TARGET_INDICES]

        # Calculate MCRMSE Loss
        # 1. MSE per column (averaged over batch and sequence)
        mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))
        # 2. RMSE per column
        rmse = torch.sqrt(mse + 1e-8)  # Add epsilon for stability
        # 3. Mean of RMSEs
        loss = torch.mean(rmse)

        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before calculating MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU

            preds = model(inputs, pair_indices, pair_mask)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using the utility function (handles slicing internally)
    score = MCRMSE(all_targets, all_preds)

    return score.item()


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    Returns ids and full predictions (Batch, 107, 5).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["ids"]

            preds = model(inputs, pair_indices, pair_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids


def run_training():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WORKING_DIR, exist_ok=True)
    model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

    # 2. Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=DEBUG, debug_subset_size=DEBUG_SUBSET_SIZE
    )

    # 3. Model
    print("Initializing Model...")
    model = DeepPostNormBiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Training Loop
    best_val_score = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Printed with full precision
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  >>> New Best Model Saved (Score: {best_val_score})")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation Score: {best_val_score}")

    # 6. Inference
    print("Generating Submission...")
    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    preds, ids = predict_test(model, test_loader, device)

    # 7. Format Submission
    # We need to flatten the predictions to match the submission format:
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 5)

        for seq_pos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(TARGET_COLS):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure column order
    cols = ["id_seqpos"] + TARGET_COLS
    submission_df = submission_df[cols]

    # Save
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    run_training()
