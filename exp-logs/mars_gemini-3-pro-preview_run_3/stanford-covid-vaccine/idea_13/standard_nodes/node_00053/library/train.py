import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config, setup_reproducibility
from library.utils import get_device, mcrmse_loss, compute_mcrmse_numpy
from library.data import get_dataloaders
from library.model import GatedSpatialConvBiGRU


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 5)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (B, 107, 5)
        outputs = model(inputs, pair_indices)

        # Slice outputs to match targets (first 68 positions)
        outputs_scored = outputs[:, : Config.SEQ_SCORED, :]

        # Compute loss (Unweighted MCRMSE on all 5 targets)
        loss = mcrmse_loss(outputs_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Crucial for RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VAL)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Global Metric Aggregation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].cpu().numpy()  # Shape: (B, 68, 5)

            # Forward pass
            outputs = model(inputs, pair_indices)

            # Slice to scored length and move to CPU
            outputs_scored = outputs[:, : Config.SEQ_SCORED, :].cpu().numpy()

            all_preds.append(outputs_scored)
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Filter for the 3 scored columns for the competition metric
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Identify indices of scored columns
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    y_pred_scored = y_pred[:, :, scored_indices]
    y_true_scored = y_true[:, :, scored_indices]

    # Compute MCRMSE
    val_mcrmse = compute_mcrmse_numpy(y_pred_scored, y_true_scored)

    return val_mcrmse


def generate_submission(model, dataloader, device, submission_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass -> (B, 107, 5)
            outputs = model(inputs, pair_indices)
            outputs = outputs.cpu().numpy()

            # Store results
            # We need to predict for ALL positions (107), not just scored ones.
            for i, sample_id in enumerate(ids):
                sample_pred = outputs[i]  # (107, 5)

                # Create row identifiers: id_{sample_id}_{seqpos}
                for seqpos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"
                    ids_list.append(row_id)
                    preds_list.append(sample_pred[seqpos])

    # Create DataFrame
    cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    preds_array = np.array(preds_list)

    submission_df = pd.DataFrame(preds_array, columns=cols)
    submission_df.insert(0, "id_seqpos", ids_list)

    # Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training():
    # 1. Setup
    setup_reproducibility(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = GatedSpatialConvBiGRU(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! (MCRMSE: {best_mcrmse})")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)


if __name__ == "__main__":
    run_training()
