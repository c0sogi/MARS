import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import Config
from library.utils import probabilistic_f1


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move inputs to device
        images = batch["image"].to(device, non_blocking=True)
        images_contra = batch["image_contra"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images, images_contra)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient Clipping: Explicitly DISABLED as per requirements
        # if Config.MAX_GRAD_NORM:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Update weights
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            images_contra = batch["image_contra"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            logits = model(images, images_contra)
            loss = criterion(logits, labels)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item()
            num_batches += 1

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds).flatten()
        y_true = np.concatenate(all_labels).flatten()
        pf1 = probabilistic_f1(y_true, y_pred)
    else:
        pf1 = 0.0

    return avg_loss, pf1


def run_training(model, train_loader, val_loader, device):
    """
    Orchestrates the full training loop with Early Stopping.
    """
    # 1. Setup Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

    # 2. Setup Loss Function
    # Aggressive positive weighting for class imbalance
    pos_weight_tensor = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 3. Early Stopping Setup
    best_pf1 = -1.0
    patience = 3
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1}"
        )

        # Early Stopping Logic
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            # Save Best Model
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"New best pF1! Model saved to {Config.MODEL_CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val pF1: {best_pf1}")


def make_submission(model, test_loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    Aggregates predictions by prediction_id using Max.
    """
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()

    prediction_ids = []
    probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            images_contra = batch["image_contra"].to(device, non_blocking=True)
            ids = batch["prediction_id"]

            logits = model(images, images_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            prediction_ids.extend(ids)
            probabilities.extend(probs)

    # Create DataFrame
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": probabilities})

    # Aggregate by prediction_id (Max probability across views)
    # A patient might have multiple views (CC, MLO) mapping to the same prediction_id (Breast level)
    submission_df = df_pred.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
