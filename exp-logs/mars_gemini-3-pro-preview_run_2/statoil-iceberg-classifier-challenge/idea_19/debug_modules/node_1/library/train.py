import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss
import copy

# Import from library modules
from library.config import Config
from library.utils import set_seed, EarlyStopping
from library.data_loader import get_processed_data, get_fold_loaders, get_test_loader
from library.model import DWB_DPN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0

    for imgs, incs, labels in loader:
        imgs = imgs.to(device)
        incs = incs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs, incs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average validation loss, list of probabilities)
    """
    model.eval()
    running_loss = 0.0
    preds = []

    with torch.no_grad():
        for imgs, incs, labels in loader:
            imgs = imgs.to(device)
            incs = incs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs, incs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            batch_preds = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(batch_preds)

    return running_loss / len(loader), preds


def run_fold(fold_idx, X_train, y_train, inc_train, device):
    """
    Runs the training process for a single fold.

    Args:
        fold_idx: Index of the current fold.
        X_train, y_train, inc_train: Full training data arrays.
        device: Torch device.

    Returns:
        tuple: (Best validation loss, Validation predictions for this fold, Validation indices)
    """
    print(f"\nStarting Fold {fold_idx + 1}/{Config.N_FOLDS}")

    # Get Loaders for this fold
    train_loader, val_loader = get_fold_loaders(X_train, y_train, inc_train, fold_idx)

    # Initialize Model
    model = DWB_DPN().to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="min")

    best_val_preds = []
    best_val_loss = float("inf")

    # Identify validation indices for OOF mapping (re-creating logic from get_fold_loaders)
    # We need to know which indices belong to this validation fold to map predictions back
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    val_indices = list(skf.split(X_train, y_train))[fold_idx][1]

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Fold {fold_idx+1} Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Check Early Stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            best_val_loss = early_stopping.best_score
            # We need to reload the best state to get the predictions associated with best loss
            model.load_state_dict(early_stopping.best_model_state)
            _, best_val_preds = validate(model, val_loader, criterion, device)
            break

        # If this was the best epoch so far (tracked by early_stopping), update our return values
        if val_loss == early_stopping.best_score:
            best_val_loss = val_loss
            best_val_preds = val_preds

    # Save the best model for this fold
    save_path = os.path.join(Config.WORKING_DIR, f"dwb_dpn_fold_{fold_idx}.pth")
    torch.save(early_stopping.best_model_state, save_path)
    print(f"Best model for fold {fold_idx+1} saved to {save_path}")

    return best_val_loss, np.array(best_val_preds).flatten(), val_indices


def run_training():
    """
    Main function to run the full training pipeline:
    1. Load Data
    2. Run Stratified K-Fold Training
    3. Generate OOF Metrics
    4. Generate Test Predictions (Ensemble)
    5. Save Submission
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Data
    X_train, y_train, inc_train, X_test, inc_test, test_ids = get_processed_data(
        load_cached_data=True
    )

    # Arrays to store results
    oof_preds = np.zeros(len(X_train))
    test_preds_accum = np.zeros(len(X_test))

    # 2. Run Stratified K-Fold Training
    for fold in range(Config.N_FOLDS):
        _, val_preds, val_indices = run_fold(fold, X_train, y_train, inc_train, device)

        # Store OOF predictions
        oof_preds[val_indices] = val_preds

        # Inference on Test Set with this fold's model
        print(f"Generating test predictions for Fold {fold+1}...")

        # Load the best model state
        model = DWB_DPN().to(device)
        model_path = os.path.join(Config.WORKING_DIR, f"dwb_dpn_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        test_loader = get_test_loader(X_test, inc_test)
        fold_test_preds = []

        with torch.no_grad():
            for imgs, incs in test_loader:
                imgs = imgs.to(device)
                incs = incs.to(device)
                outputs = model(imgs, incs)
                fold_test_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        test_preds_accum += np.array(fold_test_preds).flatten()

    # 3. Generate OOF Metrics
    oof_loss = log_loss(y_train, oof_preds)
    print(f"\nOverall OOF Log Loss: {oof_loss:.10f}")

    # 4. Generate Test Predictions (Ensemble Average)
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # 5. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
