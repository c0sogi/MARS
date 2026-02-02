import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import mcrmse

# =========================================================================
# Loss Function
# =========================================================================


def masked_mse_loss(preds, targets):
    """
    Calculates MSE loss only on the scored positions (first 68).
    Args:
        preds: (Batch, Seq_Len, Num_Targets)
        targets: (Batch, Seq_Len, Num_Targets)
    """
    # Slice to the scored length defined in Config
    preds_scored = preds[:, : Config.SEQ_SCORED, :]
    targets_scored = targets[:, : Config.SEQ_SCORED, :]

    loss = nn.MSELoss()(preds_scored, targets_scored)
    return loss


# =========================================================================
# Early Stopping
# =========================================================================


class EarlyStopping:
    def __init__(self, patience=5, mode="min", save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.save_path = save_path

        if mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, score, model):
        if self.mode == "min":
            score_improved = score < self.val_score
        else:
            score_improved = score > self.val_score

        if score_improved:
            self.best_score = score
            self.val_score = score
            self.counter = 0
            self.save_checkpoint(model)
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved

    def save_checkpoint(self, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)


# =========================================================================
# Training & Validation
# =========================================================================


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        seq_inputs = batch["seq_inputs"].to(device)
        pair_dists = batch["pair_dists"].to(device)
        loop_types = batch["loop_types"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(seq_inputs, pair_dists, loop_types)

        # Calculate Loss (Masked)
        loss = masked_mse_loss(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq_inputs = batch["seq_inputs"].to(device)
            pair_dists = batch["pair_dists"].to(device)
            loop_types = batch["loop_types"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq_inputs, pair_dists, loop_types)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    # Note: mcrmse function handles slicing internally if needed,
    # but we pass full tensors and let it handle the logic based on Config.
    score = mcrmse(all_targets, all_preds, num_scored=Config.SEQ_SCORED)

    return score


# =========================================================================
# Submission Generation
# =========================================================================


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []

    # 1. Inference
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in loader:
            seq_inputs = batch["seq_inputs"].to(device)
            pair_dists = batch["pair_dists"].to(device)
            loop_types = batch["loop_types"].to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len, 3)
            preds = model(seq_inputs, pair_dists, loop_types)
            all_preds.append(preds.cpu().numpy())

    # Shape: (Num_Samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # 2. Load Test Metadata to get IDs
    # We assume the loader preserves order (shuffle=False)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)
    sample_ids = df_test["id"].values

    # 3. Construct Submission Data
    # We need to predict 5 columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # The model predicts 3: reactivity, deg_Mg_pH10, deg_Mg_50C
    # We fill the missing ones with 0.0

    # Map model output indices to submission column names
    # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
    pred_map = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 2}

    submission_rows = []
    seq_len = Config.SEQ_LENGTH  # 107

    for i, sample_id in enumerate(sample_ids):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"

            # Extract predictions
            reactivity = float(sample_preds[pos, pred_map["reactivity"]])
            deg_Mg_pH10 = float(sample_preds[pos, pred_map["deg_Mg_pH10"]])
            deg_Mg_50C = float(sample_preds[pos, pred_map["deg_Mg_50C"]])

            # Fill missing columns with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    # 4. Create DataFrame and Save
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# =========================================================================
# Main Training Loop
# =========================================================================


def fit(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Orchestrates the training process with early stopping.
    """
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, mode="min", save_path=Config.MODEL_SAVE_PATH
    )

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Early Stopping Check
        improved = early_stopping(val_score, model)
        if improved:
            print(f"  -> Model Saved! New Best Score: {val_score:.10f}")

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model for final state
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    return model
