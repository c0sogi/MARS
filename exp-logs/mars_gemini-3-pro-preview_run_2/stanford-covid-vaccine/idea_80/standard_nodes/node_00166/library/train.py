import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_dataloaders
from library.model import AS_DRN


class MCRMSELoss(nn.Module):
    """
    Differentiable Mean Columnwise Root Mean Squared Error Loss.
    Calculates loss over the full sequence length (0-107) and all targets
    to enforce Boundary Anchoring.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # inputs: (N, L, 5)
        # targets: (N, L, 5)

        # Compute MSE for each column (averaging over Batch and Sequence dimensions)
        mse = torch.mean((inputs - targets) ** 2, dim=(0, 1))

        # Compute RMSE for each column (adding epsilon for stability)
        rmse = torch.sqrt(mse + 1e-8)

        # Return the mean of column RMSEs
        return torch.mean(rmse)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch with the Iterative Refinement Loop.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass returns both final (y_2) and intermediate (y_1) predictions
        y_2, y_1 = model(inputs, partner_indices)

        # Calculate Anchored Loss
        # L_total = MCRMSE(y_2) + 0.5 * MCRMSE(y_1)
        # Loss is calculated over full sequence length (0-107)
        loss_2 = criterion(y_2, targets)
        loss_1 = criterion(y_1, targets)
        loss = loss_2 + 0.5 * loss_1

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Calculates Correct Global MCRMSE on scored positions only.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Inference: The model forward returns (y_2, y_1), we take y_2
            y_2, _ = model(inputs, partner_indices)

            all_preds.append(y_2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric using the utility function (handles slicing to scored length/cols)
    score = mcrmse_metric(all_targets, all_preds)

    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["id"]

            # Inference
            y_2, _ = model(inputs, partner_indices)

            all_preds.append(y_2.cpu().numpy())
            all_ids.extend(ids)

    # Shape: (N_samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare data for DataFrame
    # We need to flatten the predictions to one row per (id, seqpos)
    submission_data = []

    seq_len = Config.SEQ_LENGTH
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            row_preds = sample_preds[pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission_df)} rows)")


def run_training(debug=False):
    """
    Main execution function for training, validation, and submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model
    print("Initializing AS-DRN Model...")
    model = AS_DRN().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score:.6f}")

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
