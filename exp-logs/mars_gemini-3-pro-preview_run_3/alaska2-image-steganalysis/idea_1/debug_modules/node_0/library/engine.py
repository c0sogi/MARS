import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import AverageMeter, weighted_auc_score
from library.config import IDEA_DIR, SUBMISSION_PATH


def train_one_epoch(model, loader, optimizer, criterion, device, label_smoothing=0.0):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run training on.
        label_smoothing (float): Factor for label smoothing (0.0 to disable).

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        # Apply Label Smoothing manually
        # Target transformation: new_y = y * (1 - alpha) + 0.5 * alpha
        if label_smoothing > 0:
            smoothed_targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            smoothed_targets = targets

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, smoothed_targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Weighted AUC score)
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Calculate Weighted AUC
    score = weighted_auc_score(all_targets, all_preds)

    return losses.avg, score


def predict_tta(model, loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Saves the result to the submission file.

    Args:
        model: The trained model.
        loader: DataLoader for test data.
        device: Device to run inference on.
    """
    model.eval()
    results = []

    # Retrieve Image IDs from the dataset dataframe
    # We assume the loader preserves order (shuffle=False)
    dataset_df = loader.dataset.df
    image_ids = dataset_df["image_id"].values

    current_idx = 0

    with torch.no_grad():
        for inputs, _ in loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)

            # TTA View 1: Original
            out0 = torch.sigmoid(model(inputs))

            # TTA View 2: 90 degrees rotation
            inputs_90 = torch.rot90(inputs, 1, [2, 3])
            out90 = torch.sigmoid(model(inputs_90))

            # TTA View 3: 180 degrees rotation
            inputs_180 = torch.rot90(inputs, 2, [2, 3])
            out180 = torch.sigmoid(model(inputs_180))

            # TTA View 4: 270 degrees rotation
            inputs_270 = torch.rot90(inputs, 3, [2, 3])
            out270 = torch.sigmoid(model(inputs_270))

            # Average predictions across all views
            avg_preds = (out0 + out90 + out180 + out270) / 4.0
            avg_preds = avg_preds.cpu().numpy().flatten()

            # Map predictions to Image IDs
            batch_ids = image_ids[current_idx : current_idx + batch_size]

            for img_id, score in zip(batch_ids, avg_preds):
                results.append({"Id": img_id, "Label": score})

            current_idx += batch_size

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save submission
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    return submission_df


def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    early_stopping_patience,
    label_smoothing,
):
    """
    Main driver function to run the training loop, validation, early stopping, and inference.
    """
    # Initialize Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(IDEA_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, label_smoothing
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_score}"
        )

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights for final prediction
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} with AUC {best_score}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model saved. Using current model weights.")

    # Generate Submission
    print("Generating predictions on test set with TTA...")
    predict_tta(model, test_loader, device)
    print(f"Submission saved to {SUBMISSION_PATH}")
