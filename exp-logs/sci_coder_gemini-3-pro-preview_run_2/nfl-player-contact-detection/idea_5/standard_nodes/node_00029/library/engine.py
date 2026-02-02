import os
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer instance.
        criterion: The loss function (e.g., FocalLoss).
        device: The device to run on (cpu or cuda).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        running_loss += loss.item() * X.size(0)
        count += X.size(0)

    return running_loss / count if count > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the provided dataloader.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation/test data.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, true_labels, predicted_probabilities)
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(X)
            loss = criterion(logits, y)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * X.size(0)
            count += X.size(0)

            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate results
    if len(all_targets) > 0:
        y_true = np.concatenate(all_targets).flatten()
        y_probs = np.concatenate(all_probs).flatten()
    else:
        y_true = np.array([])
        y_probs = np.array([])

    return avg_loss, y_true, y_probs


def find_best_threshold(y_true, y_probs):
    """
    Performs a grid search to find the threshold that maximizes MCC.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Define search space from Config
    thresholds = np.arange(
        Config.THRESHOLD_SEARCH_START,
        Config.THRESHOLD_SEARCH_END,
        Config.THRESHOLD_SEARCH_STEP,
    )

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Runs the training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model weights.

    Returns:
        tuple: (best_model_state_dict, best_mcc_score)
    """
    best_mcc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, val_y, val_probs = evaluate(model, val_loader, criterion, device)

        # Optimize Threshold
        thresh, val_mcc = find_best_threshold(val_y, val_probs)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCC: {val_mcc} | Best Thresh: {thresh}"
        )

        # Early Stopping Logic
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best weights into model
    model.load_state_dict(best_model_wts)
    return model, best_mcc


def inference(model, test_loader, test_df, device, threshold):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        test_df: DataFrame containing test metadata (must match loader order).
        device: Device.
        threshold: Decision threshold for binary classification.

    Returns:
        None
    """
    model.eval()
    all_probs = []

    # Generate probabilities
    with torch.no_grad():
        for X in test_loader:
            # Handle case where loader returns (X, y) or just X
            if isinstance(X, list) or isinstance(X, tuple):
                X = X[0]

            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    # Flatten
    y_probs = np.concatenate(all_probs).flatten()

    # Apply threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Create submission DataFrame
    # We assume test_df is aligned with test_loader (guaranteed by data pipeline)
    submission = pd.DataFrame({"contact_id": test_df["contact_id"], "contact": y_pred})

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with {len(submission)} rows.")
