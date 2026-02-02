import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, calculate_log_loss
from library.data_loader import get_loaders, get_test_loader
from library.model import ATSICNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # (B,) -> (B, 1)

        optimizer.zero_grad()

        # Forward pass: Model takes both image and angle
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and log loss.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    y_pred = np.vstack(preds_list)
    y_true = np.vstack(labels_list)

    # Calculate Log Loss metric
    # y_pred are probabilities, y_true are 0/1
    metric_score = calculate_log_loss(y_true, y_pred)

    return epoch_loss, metric_score


def run_kfold():
    """
    Runs 5-Fold Cross-Validation training.
    Saves the best model for each fold.
    """
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_scores = []

    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold in range(Config.NUM_FOLDS):
        print(f"\n--- Fold {fold} ---")

        # Get data loaders for this fold
        train_loader, val_loader = get_loaders(fold, load_cached_data=True)

        # Initialize Model
        model = ATSICNN().to(device)

        # Optimizer: AdamW with constant LR
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss: BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping variables
        best_val_logloss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_logloss = validate(model, val_loader, criterion, device)

            print(
                f"Fold {fold} Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val LogLoss: {val_logloss:.10f}"
            )

            # Check for improvement (monitor Log Loss)
            if val_logloss < best_val_logloss:
                best_val_logloss = val_logloss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Fold {fold} Best LogLoss: {best_val_logloss:.10f}")
        fold_scores.append(best_val_logloss)

    avg_score = np.mean(fold_scores)
    print(f"\nCross-Validation Complete. Average LogLoss: {avg_score:.10f}")


def generate_submission():
    """
    Loads trained models from all folds, predicts on the test set,
    averages the predictions, and creates the submission file.
    """
    print("\nGenerating Submission...")

    device = torch.device(Config.DEVICE)
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Get Test Loader
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Array to store accumulated probabilities
    # Shape: (N_test, 1)
    avg_preds = np.zeros((len(test_ids), 1), dtype=np.float32)

    # Iterate over each fold
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")

        # Load Model
        model = ATSICNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)

                # Forward pass
                outputs = model(images, angles)

                # Sigmoid to get probability
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        # Stack predictions for this fold
        fold_preds_arr = np.vstack(fold_preds)

        # Add to accumulator
        avg_preds += fold_preds_arr

    # Average predictions
    avg_preds /= Config.NUM_FOLDS

    # Flatten to 1D array
    avg_preds = avg_preds.flatten()

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main entry point to execute the full pipeline.
    """
    run_kfold()
    generate_submission()
