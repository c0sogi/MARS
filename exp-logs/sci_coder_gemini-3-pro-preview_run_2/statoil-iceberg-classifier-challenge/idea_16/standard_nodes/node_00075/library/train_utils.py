import os
import copy
import torch
import torch.nn as nn
from library.config import Config


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model weights using copy.deepcopy.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Deep copy the model state to preserve the best weights in memory
        self.best_model_state = copy.deepcopy(model.state_dict())

        # Save to disk
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: The training dataloader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, angles, labels) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        # CSNet takes both image and angle as input
        outputs = model(images, angles)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average validation loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, labels in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_fold(
    model, train_loader, val_loader, optimizer, scheduler, device, config, fold_idx
):
    """
    Orchestrates the training for a single fold, including Early Stopping and Scheduling.

    Args:
        model: The CSNet model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Computation device.
        config: Configuration object.
        fold_idx: Index of the current fold (for saving paths).

    Returns:
        model: The model with the best weights loaded.
        float: The best validation loss achieved.
    """
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(config.MODEL_DIR, f"csnet_fold_{fold_idx}.pth")

    early_stopping = EarlyStopping(
        patience=config.PATIENCE, verbose=True, path=checkpoint_path
    )

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics as requested
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Scheduler step (ReduceLROnPlateau expects val_loss)
        if scheduler:
            scheduler.step(val_loss)

        # Early Stopping check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    if early_stopping.best_model_state is not None:
        print(
            f"Restoring best model weights from epoch with Val Loss: {early_stopping.val_loss_min}"
        )
        model.load_state_dict(early_stopping.best_model_state)

    return model, early_stopping.val_loss_min
