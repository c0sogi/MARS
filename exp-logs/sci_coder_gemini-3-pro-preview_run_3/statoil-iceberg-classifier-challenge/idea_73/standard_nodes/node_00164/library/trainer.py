import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.model import ADSICNN
from library.data_loader import load_data, get_dataloaders, get_test_loader
from library.utils import set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for imgs, angles, labels in loader:
        imgs = imgs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()
        outputs = model(imgs, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and log loss metric.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for imgs, angles, labels in loader:
            imgs = imgs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(outputs)
            preds_list.extend(probs.cpu().numpy())
            targets_list.extend(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Calculate Log Loss
    y_true = np.array(targets_list).flatten()
    y_pred = np.array(preds_list).flatten()
    metric_score = log_loss(y_true, y_pred)

    return epoch_loss, metric_score


def run_training(load_cached_data=True, epochs=None, sample_size=None):
    """
    Main function to run 5-Fold Cross-Validation training and generate submission.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        epochs (int, optional): Override Config.EPOCHS.
        sample_size (int, optional): Limit dataset size for debugging.
    """
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE

    num_epochs = epochs if epochs is not None else Config.EPOCHS

    # Load Data
    # load_data handles caching internally based on the flag
    X, y, angles, ids_train, X_test, angle_test, ids_test = load_data(
        load_cached_data=load_cached_data
    )

    # Debugging: Slice data if sample_size is provided
    if sample_size is not None:
        print(f"DEBUG: Limiting dataset to {sample_size} samples.")
        X = X[:sample_size]
        y = y[:sample_size]
        angles = angles[:sample_size]
        ids_train = ids_train[:sample_size]
        # Also slice test data to speed up inference during debug
        X_test = X_test[:sample_size]
        angle_test = angle_test[:sample_size]
        ids_test = ids_test[:sample_size]

    # Initialize K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Accumulator for test set predictions
    test_preds_accum = np.zeros(len(X_test))

    print(f"Starting training on device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_dataloaders(X, y, angles, train_idx, val_idx)

        # Initialize Model
        model = ADSICNN().to(device)

        # Optimizer and Loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_metric = validate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val LogLoss: {val_metric}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Inference on Test Set
        # We pass the full training angles to impute test angles correctly
        test_loader = get_test_loader(X_test, angle_test, angles)

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs = imgs.to(device)
                angs = angs.to(device)
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy())

        # Accumulate predictions
        fold_preds = np.array(fold_preds).flatten()
        test_preds_accum += fold_preds

    # Average predictions across folds
    avg_preds = test_preds_accum / Config.N_FOLDS

    # Generate Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
