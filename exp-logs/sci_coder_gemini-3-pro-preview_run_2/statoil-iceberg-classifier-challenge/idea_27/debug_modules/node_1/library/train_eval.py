import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, accuracy_score

from library.model import SWDINet
from library.data_loader import get_fold_loaders
from library.utils import seed_everything


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Strictly preserves the best model weights.
    """

    def __init__(self, patience=10, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        """Restores the best model weights."""
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)


def train_one_epoch(model, loader, optimizer, criterion, device, debug=False):
    """
    Handles the training of one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for i, (images, angles, labels) in enumerate(loader):
        if debug and i >= 2:  # Limit to 2 batches in debug mode
            break

        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect metrics
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_targets) if all_targets else 0
    epoch_acc = accuracy_score(all_targets, np.round(all_preds)) if all_targets else 0

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, debug=False):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for i, (images, angles, labels) in enumerate(loader):
            if debug and i >= 2:
                break

            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels.unsqueeze(1))

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_targets) if all_targets else 0
    epoch_acc = accuracy_score(all_targets, np.round(all_preds)) if all_targets else 0

    return epoch_loss, epoch_acc, np.array(all_preds)


def predict(model, loader, device, debug=False):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if debug and i >= 2:
                break

            # Handle different return signatures (Dataset vs DataLoader behavior)
            if len(batch) == 2:
                images, angles = batch
            else:
                images, angles, _ = batch

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)

    return np.array(all_preds)


def run_training(
    epochs=50,
    batch_size=32,
    patience=10,
    seed=42,
    output_dir="./submission",
    debug=False,
):
    """
    Main execution function.
    1. Runs Stratified 5-Fold CV.
    2. Trains SWDI-Net with independent scaling per fold.
    3. Generates ensemble predictions.
    4. Saves submission file.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
        seed (int): Random seed.
        output_dir (str): Directory to save submission.
        debug (bool): If True, runs a truncated version for debugging.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Accumulator for test predictions across folds
    test_preds_accum = None
    ids_test_final = None

    n_splits = 5

    print(f"Starting {n_splits}-Fold Cross-Validation...")

    for fold in range(n_splits):
        print(f"\n=== FOLD {fold} ===")

        # Get Fold DataLoaders (Handles independent scaling)
        train_loader, val_loader, test_loader, ids_test = get_fold_loaders(
            fold_idx=fold, n_splits=n_splits, batch_size=batch_size, seed=seed
        )

        # Initialize accumulator on first fold
        if ids_test_final is None:
            ids_test_final = ids_test
            # If debug, we might not process full test set, so we handle shape dynamically or assume full run
            if not debug:
                test_preds_accum = np.zeros((len(ids_test), 1))
            else:
                # In debug mode, we might not get full predictions, placeholder
                pass

        # Initialize Model
        model = SWDINet().to(device)

        # Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        # Training Loop
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, debug=debug
            )
            val_loss, val_acc, val_probs = validate(
                model, val_loader, criterion, device, debug=debug
            )

            scheduler.step(val_loss)

            # Print full precision metrics
            print(
                f"Fold {fold} Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best weights
        early_stopping.load_best_weights(model)

        # Final Validation Check
        best_val_loss, best_val_acc, _ = validate(
            model, val_loader, criterion, device, debug=debug
        )
        print(f"Fold {fold} Best Val Loss: {best_val_loss}")

        # Predict on Test (Ensemble)
        if not debug:
            fold_test_preds = predict(model, test_loader, device, debug=debug)
            test_preds_accum += fold_test_preds.reshape(-1, 1)

        # Clean up
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    if not debug:
        # Average predictions
        avg_test_preds = test_preds_accum / n_splits

        # Save Submission
        os.makedirs(output_dir, exist_ok=True)
        sub_df = pd.DataFrame(
            {"id": ids_test_final, "is_iceberg": avg_test_preds.flatten()}
        )

        # Clip for log loss safety
        sub_df["is_iceberg"] = sub_df["is_iceberg"].clip(0.001, 0.999)

        submission_path = os.path.join(output_dir, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print("Debug run complete. No submission generated.")
