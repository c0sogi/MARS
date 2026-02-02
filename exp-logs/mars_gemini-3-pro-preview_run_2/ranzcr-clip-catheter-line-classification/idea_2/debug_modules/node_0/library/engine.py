import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_score


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss and OneCycleLR.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (OneCycleLR).
        dataloader: The training dataloader.
        device: The computing device (cpu or cuda).
        epoch: The current epoch number.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # OneCycleLR steps every batch
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch+1} Train Loss: {epoch_loss}")

    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The computing device.

    Returns:
        tuple: (average validation loss, average AUC score)
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu().numpy())
            valid_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)

    print(f"Validation Loss: {epoch_loss}")

    # Calculate AUC using the utility function
    # This function prints individual column AUCs and returns the average
    avg_auc = get_score(valid_targets, preds)

    return epoch_loss, avg_auc


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Main training loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Device.
        epochs: Total number of epochs.

    Returns:
        float: The best validation AUC achieved.
    """
    best_auc = -1.0
    patience = Config.EARLY_STOPPING_PATIENCE
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, device)

        # Early Stopping and Model Checkpointing
        if val_auc > best_auc:
            print(
                f"Validation AUC improved from {best_auc} to {val_auc}. Saving model to {Config.MODEL_SAVE_PATH}..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation AUC did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.

    Args:
        model: The PyTorch model.
        dataloader: Dataloader for the test set.
        device: Device.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, dtype=torch.float)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def generate_submission(model, test_loader, device):
    """
    Generates the submission CSV file using the trained model.

    Args:
        model: The trained PyTorch model.
        test_loader: Dataloader for the test set.
        device: Device.
    """
    print("Generating predictions for submission...")

    # Load test metadata to get the correct order of StudyInstanceUIDs
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Generate predictions
    predictions = predict(model, test_loader, device)

    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)

    # Insert StudyInstanceUID as the first column
    submission_df.insert(0, "StudyInstanceUID", df_test["StudyInstanceUID"])

    # Ensure output directory exists
    output_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
