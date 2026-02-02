import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, get_device, mcrmse_loss, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel


def train_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices)

        # Calculate loss (MCRMSE on all 5 columns)
        loss = mcrmse_loss(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before calculating the metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate to form global tensors
    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate metric (handles slicing and column filtering internally)
    score = metric_mcrmse(global_preds, global_targets)

    return score


def inference(model, loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass
            preds = model(inputs, pair_indices)  # Shape: (B, 107, 5)

            # Store results
            preds_list.append(preds.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape: (107, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save to CSV
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    """
    Main execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = RNAModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    inference(model, test_loader, device)
