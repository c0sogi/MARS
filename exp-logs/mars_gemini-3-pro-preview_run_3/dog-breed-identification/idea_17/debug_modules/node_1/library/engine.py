import torch
import torch.nn as nn
import numpy as np
import os


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Device to train on (cpu or cuda).
        epoch: Current epoch number (for logging).

    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    print(f"Epoch {epoch} Training Loss: {epoch_loss} Accuracy: {epoch_acc}")

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Device to evaluate on.

    Returns:
        tuple: (val_loss, val_acc)
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += batch_size

    val_loss = running_loss / total_samples
    val_acc = correct_predictions / total_samples

    print(f"Validation Loss: {val_loss} Accuracy: {val_acc}")

    return val_loss, val_acc


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Specifically, averages predictions from the original image and a horizontally flipped version.

    Args:
        model: The PyTorch model.
        dataloader: Test DataLoader.
        device: Device to predict on.

    Returns:
        tuple: (probabilities_numpy_array, ids_list)
    """
    model.eval()

    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Forward pass: Original
            outputs_orig = model(images)
            probs_orig = torch.softmax(outputs_orig, dim=1)

            # 2. Forward pass: Horizontal Flip
            # Tensor is (N, C, H, W), so we flip on dim 3
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    final_probs = np.concatenate(all_probs, axis=0)
    return final_probs, all_ids


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
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
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

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

        # Handle cases where model might be wrapped in DataParallel
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
