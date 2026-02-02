import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.loss_metrics import MCRMSELoss, calculate_metric_mcrmse
from library.model import DeepStabilizedBiGRU
from library.data_utils import get_dataloaders, seed_everything


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        bpp_masks = batch["bpp_masks"].to(device)
        targets = batch["targets"].to(device)
        target_masks = batch["target_masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, bpp_indices, bpp_masks)

        # Loss calculation
        loss = criterion(outputs, targets, target_masks)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch["targets"].to(device)
            target_masks = batch["target_masks"].to(device)

            outputs = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(target_masks.cpu())

    # Concatenate for global metric calculation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate MCRMSE using the library function which handles slicing and column filtering
    metric = calculate_metric_mcrmse(all_preds, all_targets, all_masks)

    return metric


def train_model():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 2. Model
    model = DeepStabilizedBiGRU().to(device)

    # 3. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = MCRMSELoss()

    # 4. Training Loop with Early Stopping
    best_metric = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metric = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_metric}"
        )

        # Early Stopping Logic
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! (Metric: {best_metric})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_metric}")


def generate_submission():
    """
    Generates submission file using the best trained model.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Load Model
    model = DeepStabilizedBiGRU().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        print(
            f"Error: Model file {Config.MODEL_PATH} not found. Cannot generate submission."
        )
        return

    print(f"Loading model from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Inference
    all_ids = []
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # 4. Format Submission
    # We need to flatten the predictions to match the id_seqpos format
    # Rows: N_samples * 107
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_rows = []
    target_cols = Config.TARGET_COLS  # The 5 columns in order

    print("Formatting submission...")
    for i, sample_id in enumerate(all_ids):
        preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
