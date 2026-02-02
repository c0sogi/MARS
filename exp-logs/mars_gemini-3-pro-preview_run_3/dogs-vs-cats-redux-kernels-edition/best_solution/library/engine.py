import torch
import torch.nn as nn
import numpy as np
from library import utils


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): Device to use ('cpu' or 'cuda').
        criterion (torch.nn.Module): The loss function.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # Ensure labels are float and have the correct shape [Batch, 1] for BCEWithLogitsLoss
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate loss (loss.item() is mean of batch usually, so multiply by batch size)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation data loader.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        float: The Log Loss on the validation set.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Store predictions and targets
            # Ensure tensors are on CPU before converting to numpy
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    # Calculate Log Loss
    # utils.calculate_log_loss handles clipping and metric calculation
    val_loss = utils.calculate_log_loss(targets, preds)
    return val_loss


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, min_delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        utils.save_checkpoint(state, self.path)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    scheduler=None,
    checkpoint_name="checkpoint.pth",
):
    """
    Runs the training loop for a specified number of epochs with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Device string.
        epochs: Max epochs.
        scheduler: Learning rate scheduler (optional).
        checkpoint_name: Filename to save the best model.

    Returns:
        model: The model with the best weights loaded.
    """
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=3, path=checkpoint_name)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate(model, val_loader, device)

        if scheduler:
            # Handle schedulers that require metrics (e.g., ReduceLROnPlateau)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        early_stopping(val_loss, model, optimizer)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    print(f"Loading best model weights from {checkpoint_name}")
    utils.load_checkpoint(checkpoint_name, model=model, device=device)

    return model
