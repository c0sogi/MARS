import torch
import numpy as np
from library import utils
from library import config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to use for training.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, targets in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Model expects (x, angle)
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (str): The device to use for evaluation.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, targets in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            # Accumulate loss
            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience, fold_idx, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            fold_idx (int): Current fold index for saving checkpoints.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.fold_idx = fold_idx
        self.delta = delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, epoch):
        """
        Updates the early stopping state and saves the model if improvement is found.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save state.
            epoch (int): The current epoch number.
        """
        is_best = False

        if val_loss < (self.best_loss - self.delta):
            self.best_loss = val_loss
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # Prepare state dictionary
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": self.best_loss,
        }

        # Save checkpoint using utility function
        # This saves 'checkpoint_fold_X.pth' every time, and copies to
        # 'model_best_fold_X.pth' if is_best is True.
        utils.save_checkpoint(state, is_best, self.fold_idx)
