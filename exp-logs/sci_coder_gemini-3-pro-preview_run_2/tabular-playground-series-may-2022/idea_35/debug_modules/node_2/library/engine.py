import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(continuous, categorical)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    avg_loss = running_loss / count
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device).view(-1, 1)

            outputs = model(continuous, categorical)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    # Check if there is more than one class to avoid errors
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = 0.5

    return avg_loss, auc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
):
    """
    Orchestrates the training process with Early Stopping and Model Checkpointing.
    """
    print(f"Starting training on device: {device}")

    best_auc = -float("inf")
    patience_counter = 0

    # Ensure working directory exists for saving the model
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = evaluate(model, val_loader, device)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load best model state for future use
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    return best_auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)

            outputs = model(continuous, categorical)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def generate_submission(model, dataloader, test_ids, device):
    """
    Generates predictions and saves them to the submission file.
    """
    print("Generating predictions for test set...")
    predictions = predict(model, dataloader, device)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
