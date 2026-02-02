import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Ensure targets are (Batch, 1) for BCEWithLogitsLoss
        targets = targets.view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reshape targets
            targets = targets.view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Calculate AUC
    # Handle edge case where only one class is present in the batch/subset
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def train_model(model, train_loader, val_loader, device):
    """
    Main training loop with Early Stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print("Validation AUC improved. Model saved.")
        else:
            patience_counter += 1
            print(
                f"No improvement. EarlyStopping counter: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(ids, probs):
                results.append({"BraTS21ID": pid, "MGMT_value": prob})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
