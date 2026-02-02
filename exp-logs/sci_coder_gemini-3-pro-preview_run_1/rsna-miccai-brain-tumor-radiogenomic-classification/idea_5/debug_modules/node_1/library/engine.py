import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import SiameseEfficientNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)  # Reshape to (Batch, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)

    # Calculate AUC
    # Handle edge case where batch might only have one class
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    # Sanitize undefined metrics (nan) to ensure control flow works
    if np.isnan(auc_score):
        auc_score = 0.5

    return total_loss, auc_score


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Remove stale model artifact to prevent loading incompatible weights if training fails
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    # Initialize Model
    model = SiameseEfficientNet(pretrained=Config.PRETRAINED)
    model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def predict_and_submit(test_loader, test_dataset_df):
    """
    Generates predictions for the test set and creates the submission file.
    """
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = SiameseEfficientNet(
        pretrained=False
    )  # Pretrained weights not needed for loading state_dict

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("No trained model found. Skipping prediction.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())

    # Create Submission DataFrame
    # We rely on the order of the loader matching the dataframe, which is standard for PyTorch non-shuffled loaders
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_dataset_df["BraTS21ID"], "MGMT_value": predictions}
    )

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
