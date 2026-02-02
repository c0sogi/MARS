import torch
import numpy as np
import pandas as pd
import os
from library.config import MODEL_PATH, THRESHOLD, TEST_META_PATH
from library.utils import calculate_f1_score


def train_epoch(model, dataloader, optimizer, criterion, device, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (str): Device to run training on ('cuda' or 'cpu').
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for i, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(
    model,
    dataloader,
    criterion,
    device,
    feature_engineer,
    threshold=THRESHOLD,
    max_batches=None,
):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (Loss): Loss function.
        device (str): Device to run evaluation on.
        feature_engineer (FeatureEngineer): Object to decode binary labels to strings.
        threshold (float): Probability threshold for classification.
        max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        tuple: (Average Validation Loss, Mean F1-Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds_binary = []
    all_targets_binary = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Sigmoid and Threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            # Store predictions and targets for metric calculation
            # Move to CPU to save GPU memory
            all_preds_binary.append(preds.cpu().numpy())
            all_targets_binary.append(targets.int().cpu().numpy())

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate F1 Score on the full validation set
    if len(all_preds_binary) > 0:
        all_preds_binary = np.vstack(all_preds_binary)
        all_targets_binary = np.vstack(all_targets_binary)

        # Convert binary matrices back to list of space-delimited strings
        pred_strings = feature_engineer.inverse_transform_labels(all_preds_binary)
        target_strings = feature_engineer.inverse_transform_labels(all_targets_binary)

        val_f1 = calculate_f1_score(target_strings, pred_strings)
    else:
        val_f1 = 0.0

    return val_loss, val_f1


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    feature_engineer,
    max_batches_per_epoch=None,
):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (str): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        feature_engineer (FeatureEngineer): For decoding labels during validation.
        max_batches_per_epoch (int, optional): Limit batches for debugging.

    Returns:
        nn.Module: The trained model (loaded with best weights).
    """
    best_f1 = -1.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_batches_per_epoch,
        )

        # Validate
        val_loss, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            feature_engineer,
            max_batches=max_batches_per_epoch,
        )

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Early Stopping Check
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Validation F1 improved. Model saved to {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement in F1. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation F1: {best_f1}")

    # Load best model weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    return model


def generate_submission(
    model,
    test_loader,
    device,
    feature_engineer,
    threshold=THRESHOLD,
    submission_file=None,
):
    """
    Generates predictions for the test set and saves to a CSV file.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device.
        feature_engineer (FeatureEngineer): For decoding binary predictions.
        threshold (float): Decision threshold.
        submission_file (str): Path to save the submission CSV.
    """
    print("Generating predictions for test set...")
    model.eval()
    all_preds_binary = []

    with torch.no_grad():
        for inputs in test_loader:
            # Test loader in dataset.py returns only inputs (x)
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds_binary.append(preds.cpu().numpy())

    if len(all_preds_binary) > 0:
        all_preds_binary = np.vstack(all_preds_binary)

        # Convert to strings
        print("Decoding predictions to tags...")
        pred_strings = feature_engineer.inverse_transform_labels(all_preds_binary)

        if submission_file:
            # Load Test IDs from metadata to ensure alignment
            print(f"Loading test IDs from {TEST_META_PATH}...")
            df_test = pd.read_csv(TEST_META_PATH)

            if len(df_test) != len(pred_strings):
                raise ValueError(
                    f"Mismatch: Test metadata has {len(df_test)} rows, but generated {len(pred_strings)} predictions."
                )

            submission_df = pd.DataFrame({"Id": df_test["Id"], "Tags": pred_strings})

            print(f"Saving submission to {submission_file}...")
            submission_df.to_csv(submission_file, index=False)

    return pred_strings
