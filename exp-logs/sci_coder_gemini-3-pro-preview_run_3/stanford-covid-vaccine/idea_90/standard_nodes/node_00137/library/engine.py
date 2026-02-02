import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import MCRMSELoss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number.

    Returns:
        avg_loss: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = MCRMSELoss()
    count = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices, pair_masks)

        # Loss calculation on all 5 targets (scoring_only=False)
        # The loss function handles slicing to Config.PRED_LEN (68) internally
        loss = criterion(outputs, targets, scoring_only=False)

        # Backward pass
        loss.backward()

        # Mandatory Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        count += 1

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Global Aggregation.
    Calculates MCRMSE only on the scored columns and positions.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.

    Returns:
        val_score: The MCRMSE score on the scored columns/positions.
    """
    model.eval()
    criterion = MCRMSELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices, pair_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Global Aggregation: Concatenate tensors for the full validation set
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metric on scored columns only (scoring_only=True)
        val_score = criterion(all_preds, all_targets, scoring_only=True).item()
    else:
        val_score = 0.0

    return val_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=10,
):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Patience for early stopping.
    """
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_score = validate(model, val_loader, device)

        # Step the scheduler (Cosine Annealing usually stepped per epoch)
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with score: {best_score}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def generate_submission(
    model, test_loader, device, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The PyTorch model (should be loaded with best weights).
        test_loader: Test DataLoader.
        device: Device.
        submission_path: Path to save the CSV.
    """
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]

            # Forward pass
            outputs = model(inputs, pair_indices, pair_masks)
            # outputs shape: (Batch, 107, 5)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    if not preds_list:
        print("No predictions generated.")
        return

    # Concatenate all predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    # We need to flatten: (Total_Samples * 107) rows
    flat_ids = []
    flat_preds = []

    seq_len = all_preds.shape[1]  # 107

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)
        for seq_pos in range(seq_len):
            id_seqpos = f"{sample_id}_{seq_pos}"
            flat_ids.append(id_seqpos)
            flat_preds.append(sample_preds[seq_pos])

    flat_preds = np.array(flat_preds)

    # Create DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Target cols order matches Config.TARGET_COLS
    submission_df = pd.DataFrame(flat_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", flat_ids)

    # Save to CSV
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
