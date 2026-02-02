import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

from library import config
from library.model import KinematicMLP, FocalLoss
from library.dataset import NFLContactDataset


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(dataloader, model, optimizer, device):
    """
    Performs backpropagation for one epoch using Focal Loss.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The KinematicMLP model.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Cite solution_lesson_node_00002: Using Focal Loss to calibrate probabilities
    loss_fn = FocalLoss(
        alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, reduction="mean"
    )

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)
        batch_size = features.size(0)

        optimizer.zero_grad()

        outputs = model(features)

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def validate(dataloader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The KinematicMLP model.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (average_loss, all_predictions, all_targets)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    loss_fn = FocalLoss(
        alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, reduction="mean"
    )

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            batch_size = features.size(0)

            outputs = model(features)

            loss = loss_fn(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    return avg_loss, all_preds, all_targets


def optimize_threshold(y_true, y_pred_probs):
    """
    Finds the probability threshold that maximizes the Matthews Correlation Coefficient.

    Args:
        y_true: Numpy array of ground truth labels.
        y_pred_probs: Numpy array of predicted probabilities.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    thresholds = np.arange(0.1, 0.96, 0.05)
    best_mcc = -1.0
    best_thresh = 0.5

    for t in thresholds:
        binary_preds = (y_pred_probs > t).astype(int)

        # Calculate MCC only if there's variance in predictions to avoid errors
        if len(np.unique(binary_preds)) > 1:
            mcc = matthews_corrcoef(y_true, binary_preds)
        else:
            mcc = 0.0

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    return best_thresh, best_mcc


def train(debug=False):
    """
    Main training routine. Handles data loading, model initialization,
    training loop, validation, threshold optimization, and early stopping.

    Args:
        debug (bool): If True, uses a subset of data for faster debugging.

    Returns:
        float: The optimal threshold determined from the validation set.
    """
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Starting training on {device}...")

    # 1. Load Data
    print("Loading training dataset...")
    train_dataset = NFLContactDataset(split="train", load_cached_data=True, debug=debug)
    print("Loading validation dataset...")
    val_dataset = NFLContactDataset(
        split="validation", load_cached_data=True, debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 2. Initialize Model
    input_dim = train_dataset.features.shape[1]
    model = KinematicMLP(input_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_threshold = 0.5

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        val_loss, val_preds, val_targets = validate(val_loader, model, device)

        # Optimize threshold based on current epoch validation
        curr_best_thresh, curr_best_mcc = optimize_threshold(val_targets, val_preds)

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val MCC: {curr_best_mcc}")
        print(f"Best Threshold (Epoch): {curr_best_thresh}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = curr_best_thresh
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")
    print(f"Optimal Threshold: {best_threshold}")
    return best_threshold


def predict(threshold, debug=False):
    """
    Generates predictions for the test set using the trained model and
    the provided threshold, then saves the submission file.

    Args:
        threshold (float): Decision threshold for binary classification.
        debug (bool): If True, processes a subset of data.
    """
    set_seed(config.SEED)
    device = config.DEVICE

    print("Loading test dataset...")
    test_dataset = NFLContactDataset(split="test", load_cached_data=True, debug=debug)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    input_dim = test_dataset.features.shape[1]
    model = KinematicMLP(input_dim).to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print(f"Generating predictions with threshold {threshold}...")
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            contact_ids = batch["contact_id"]

            outputs = model(features)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(contact_ids)

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
    else:
        all_preds = np.array([])

    # Apply threshold to get binary predictions
    binary_preds = (all_preds > threshold).astype(int)

    # Create submission dataframe
    df_sub = pd.DataFrame({"contact_id": all_ids, "contact": binary_preds.flatten()})

    # Save submission
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE_PATH} with {len(df_sub)} rows.")


def run_pipeline(debug=False):
    """
    Executes the full pipeline: Train -> Predict -> Submit.
    """
    best_threshold = train(debug=debug)
    predict(best_threshold, debug=debug)
