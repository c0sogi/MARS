import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library import config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Ensure shapes match for BCEWithLogitsLoss (logits: [B, 1], targets: [B])
        loss = criterion(logits.view(-1), targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)

        # Store predictions (sigmoid for probability) and targets for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_targets.extend(targets.cpu().numpy().flatten())
        all_preds.extend(probs)

    # Calculate average loss and AUC
    epoch_loss = running_loss / len(dataloader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in the batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits.view(-1), targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_targets.extend(targets.cpu().numpy().flatten())
            all_preds.extend(probs)

    epoch_loss = running_loss / len(dataloader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training process with early stopping.
    """
    best_auc = -float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Check (Maximize AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)

    # Load test metadata to get IDs
    # We assume the test_loader iterates sequentially over the metadata file
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    if len(all_preds) != len(df_test):
        print(
            f"Warning: Number of predictions ({len(all_preds)}) does not match metadata ({len(df_test)})"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": all_preds}
    )

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission.head())
