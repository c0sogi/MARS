import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import mcrmse_loss


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        x = batch["x"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        y = batch["y"].to(device)  # Shape: (B, 68, 5)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (B, 107, 5)
        preds = model(x, pair_indices)

        # Slice predictions to match target length (seq_scored=68)
        preds_scored = preds[:, : Config.SEQ_SCORED, :]

        # Compute loss on all 5 targets
        loss = mcrmse_loss(y, preds_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for deep RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    Aggregates all predictions before calculating the metric to avoid batch bias.
    Computes MCRMSE specifically on the 3 scored columns.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to evaluate on.

    Returns:
        float: MCRMSE score on the scored columns.
    """
    model.eval()

    all_preds = []
    all_targets = []

    # Identify indices of the scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            y = batch["y"].to(device)

            preds = model(x, pair_indices)

            # Slice to scored length
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Slice to keep only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    all_preds_filtered = all_preds[:, :, scored_indices]
    all_targets_filtered = all_targets[:, :, scored_indices]

    # Calculate global MCRMSE
    score = mcrmse_loss(all_targets_filtered, all_preds_filtered)

    return score.item()


def train_model(model, train_loader, val_loader):
    """
    Main function to run the training loop.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
    """
    device = Config.DEVICE
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE (Scored): {val_score:.6f}"
        )

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Validation Score: {best_score}")


def inference(model_class, test_loader):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_class (class): The class of the model to instantiate.
        test_loader (DataLoader): Test dataloader.
    """
    device = Config.DEVICE
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Load model
    model = model_class()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Generating predictions...")

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Predict full length (107)
            preds = model(x, pair_indices)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds)

    # Concatenate all predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seq_pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
