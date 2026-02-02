import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import get_device, calculate_log_loss, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # BCEWithLogitsLoss combines Sigmoid and BCELoss, stable for both hard and soft targets
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model outputs logits (num_classes=1)
        outputs = model(images)

        # Flatten outputs to match target shape [Batch_Size]
        outputs = outputs.view(-1)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        # loss.item() is average loss per batch. Multiply by batch size to get total.
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {avg_loss:.10f}")

    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): The device to use.

    Returns:
        float: The log loss score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # targets are already on CPU or can be moved, we need them as numpy list eventually

            outputs = model(images)
            outputs = outputs.view(-1)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Calculate metric
    loss = calculate_log_loss(all_targets, all_preds)
    print(f"Validation Log Loss: {loss:.15f}")

    return loss


def predict(model, dataloader, device, tta=Config.TTA_FLIP):
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The test dataloader.
        device (torch.device): The device to use.
        tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).

    Returns:
        list: A list of tuples (id, probability).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, img_ids in dataloader:
            images = images.to(device)

            # Forward pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits).view(-1)

            if tta:
                # Horizontal Flip TTA
                # Flip along width axis (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped).view(-1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            # Store results
            # img_ids is a tensor of IDs
            ids_np = img_ids.numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(ids_np)):
                results.append((ids_np[i], probs_np[i]))

    return results


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=3,
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
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer=None, scheduler=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler)
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
            self.save_checkpoint(val_loss, model, optimizer, scheduler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        state = {"model_state_dict": model.state_dict(), "val_loss": val_loss}
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()

        save_checkpoint(state, self.path)
        self.val_loss_min = val_loss
