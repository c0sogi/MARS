import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data, model as model_lib


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the model checkpoint.
    """

    def __init__(self, patience=7, verbose=False, path="checkpoint.pth"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, loader, criterion, optimizer, device, debug=False):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

        # Debugging: limit to a few batches
        if debug and i >= 5:
            break

    return running_loss / count


def validate(model, loader, criterion, device, debug=False):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (images, angles, labels) in enumerate(loader):
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Debugging: limit to a few batches
            if debug and i >= 5:
                break

    return running_loss / count


def run_fold(
    fold_index,
    total_folds=config.NUM_FOLDS,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    patience=config.PATIENCE,
    debug=False,
):
    """
    Runs training and validation for a single fold.

    Args:
        fold_index (int): Index of the current fold (0-based).
        total_folds (int): Total number of folds.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Patience for early stopping.
        debug (bool): If True, runs on a subset of data for fewer epochs.

    Returns:
        float: Best validation loss achieved.
    """
    utils.set_seed(config.SEED)
    device = utils.get_device()

    print(f"Starting Fold {fold_index}/{total_folds}")

    # Get DataLoaders
    train_loader, val_loader = data.get_fold_loaders(
        fold_index=fold_index,
        total_folds=total_folds,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Initialize Model
    model = model_lib.TSICNN()
    model = model.to(device)

    # Optimizer and Loss
    # Using AdamW with constant learning rate as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # Checkpoint path
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, f"model_fold_{fold_index}.pth"
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=patience, verbose=False, path=checkpoint_path
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, debug=debug
        )
        val_loss = validate(model, val_loader, criterion, device, debug=debug)

        # Print full precision metrics
        print(
            f"Fold {fold_index} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Debugging: limit epochs
        if debug and epoch >= 1:
            print("Debug mode: stopping training early.")
            break

    # Load best model weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # Final validation on best model
    final_val_loss = validate(model, val_loader, criterion, device, debug=debug)
    print(f"Fold {fold_index} Best Val Loss: {final_val_loss}")

    return final_val_loss
