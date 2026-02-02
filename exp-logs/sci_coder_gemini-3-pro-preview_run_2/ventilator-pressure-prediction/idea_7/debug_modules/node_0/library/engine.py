import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from library.config import MAX_GRAD_NORM, CACHE_DIR, SUBMISSION_DIR
from library.utils import log_metric
from library.loss_metric import compute_mae


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        preds = model(x)

        # Ensure preds shape matches targets (Batch, Seq_Len)
        if preds.dim() == 3 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        loss = criterion(preds, y, u_out)

        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and average MAE.
    """
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)

            if preds.dim() == 3 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            loss = criterion(preds, y, u_out)

            # Compute MAE (Metric)
            # compute_mae handles masking for inspiratory phase
            batch_mae = compute_mae(preds, y, u_out)

            running_loss += loss.item()
            total_mae += batch_mae
            num_batches += 1

    return running_loss / num_batches, total_mae / num_batches


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    patience,
    device,
    save_path=None,
):
    """
    Main training loop with Early Stopping and Scheduler updates.
    """
    if save_path is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        save_path = os.path.join(CACHE_DIR, "best_model.pth")

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate_one_epoch(model, val_loader, criterion, device)

        # Update Scheduler (Cosine Annealing is typically per epoch)
        if scheduler is not None:
            scheduler.step()

        # Log Metrics
        print(f"Epoch {epoch+1}/{epochs}")
        log_metric("Train Loss", train_loss)
        log_metric("Val Loss", val_loss)
        log_metric("Val MAE", val_mae)

        # Early Stopping & Checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MAE: {best_val_mae}")

    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict_and_submit(model, test_loader, device, output_path=None):
    """
    Generates predictions for the test set and saves the submission file.
    """
    if output_path is None:
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            ids = batch["ids"]

            preds = model(x)

            if preds.dim() == 3 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            # Move to CPU and flatten
            all_preds.append(preds.cpu().numpy().flatten())
            all_ids.append(ids.numpy().flatten())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    ids_arr = np.concatenate(all_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids_arr, "pressure": y_pred})

    # Sort by ID to ensure alignment with sample submission
    df_sub = df_sub.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
