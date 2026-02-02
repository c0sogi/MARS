import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import (
    ISCI_CNN,
    train_one_epoch,
    validate,
    predict_test,
    load_and_process_data,
)
from library.data_loader import get_fold_loaders, get_test_loader


def train_model(debug=False):
    """
    Executes the 5-Fold Cross-Validation training pipeline.

    Args:
        debug (bool): If True, runs a shortened training loop for debugging purposes.
    """
    set_seed(Config.SEED)

    # Load data primarily to get indices for StratifiedKFold
    # The actual data loading for training happens inside get_fold_loaders via cache
    X_train, y_train, _, _, _, _, _ = load_and_process_data(load_cached_data=True)

    if debug:
        # Use a small subset for debugging
        subset_size = 100
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        n_folds = 2
        epochs = 2
        print(f"Debug mode: Training on {subset_size} samples for {epochs} epochs.")
    else:
        n_folds = Config.NUM_FOLDS
        epochs = Config.NUM_EPOCHS

    # Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Starting Fold {fold}...")

        # Retrieve DataLoaders with leak-free imputation
        train_loader, val_loader = get_fold_loaders(train_idx, val_idx)

        # Initialize Model
        model = ISCI_CNN().to(Config.DEVICE)

        # Optimizer: AdamW with constant learning rate and weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop with Early Stopping
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss = validate(model, val_loader, criterion, Config.DEVICE)

            # Print full precision metrics
            print(
                f"Fold {fold}, Epoch {epoch}: Train Loss {train_loss}, Val Loss {val_loss}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": best_loss,
                    },
                    is_best=True,
                    fold=fold,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch} for Fold {fold}")
                break


def make_submission():
    """
    Generates predictions for the test set using the trained models from all folds.
    Saves the result to submission.csv.
    """
    set_seed(Config.SEED)

    # Get Test Loader
    test_loader, ids_test = get_test_loader()

    all_fold_preds = []

    # Iterate through all folds to load best models
    for fold in range(Config.NUM_FOLDS):
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        # Load Model
        model = ISCI_CNN().to(Config.DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict
        # predict_test returns a list of probabilities
        preds = predict_test(model, test_loader, Config.DEVICE)
        all_fold_preds.append(preds)

    if not all_fold_preds:
        raise RuntimeError("No predictions generated. Check if models were trained.")

    # Average predictions across folds
    avg_preds = np.mean(all_fold_preds, axis=0)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_FILE}")
