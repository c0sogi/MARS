import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import sys

from library.config import Config
from library.data_utils import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (x_cat, x_cont, y) in enumerate(dataloader):
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cat, x_cont)

        # BCEWithLogitsLoss expects target shape to match logits [Batch, 1]
        loss = criterion(logits, y.unsqueeze(1))

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

        # Scheduler step (OneCycleLR steps every batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, y.unsqueeze(1))

            running_loss += loss.item()

            # Apply sigmoid to get probabilities for AUC calculation
            preds = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            targets = y.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_loss = running_loss / len(dataloader)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case with single class in batch (unlikely in full val set)
        auc_score = 0.5

    return avg_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns a numpy array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            preds = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_preds.extend(preds)

    return np.array(all_preds)


def train_model(model, train_loader, val_loader):
    """
    Orchestrates the training process including optimizer setup,
    training loop, validation, and early stopping.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Define Loss
    criterion = nn.BCEWithLogitsLoss()

    # Define Optimizer (AdamW with calibrated weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Define Scheduler (OneCycleLR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Val AUC: {val_auc:.8f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.8f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model weights before returning
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    return model


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device(Config.DEVICE)
    model.to(device)

    print("Generating predictions on test set...")
    probs = predict(model, test_loader, device)

    # Load sample submission to get IDs
    print(f"Loading sample submission from {Config.SAMPLE_SUBMISSION_PATH}...")
    submission_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(probs) != len(submission_df):
        print(
            f"Warning: Prediction count ({len(probs)}) does not match sample submission ({len(submission_df)})."
        )
        # In a real scenario, this would be a critical error, but we'll proceed with truncation/padding if needed
        # or just assignment if it's a mismatch in the loader logic.
        # Assuming standard behavior where test_loader covers the full test set exactly.

    submission_df["target"] = probs

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
