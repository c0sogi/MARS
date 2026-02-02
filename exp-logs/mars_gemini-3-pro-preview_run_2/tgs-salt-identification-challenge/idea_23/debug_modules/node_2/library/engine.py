import torch
import torch.nn as nn
import numpy as np
from library.utils import calc_map_score


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Performs one epoch of supervised training.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to run on ('cuda' or 'cpu').
        loss_fn (nn.Module): Loss function (e.g., CombinedLoss).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks, depths, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, depths)

        # Calculate loss
        loss = loss_fn(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def train_student_epoch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    loss_fn_labeled,
    loss_fn_unlabeled=None,
):
    """
    Performs one epoch of semi-supervised student training (Noisy Student).
    Iterates through the labeled loader and cycles through the unlabeled loader.

    Args:
        model (nn.Module): The student model.
        labeled_loader (DataLoader): Loader for labeled data (Ground Truth).
        unlabeled_loader (DataLoader): Loader for unlabeled data (Soft Pseudo-labels).
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to run on.
        loss_fn_labeled (nn.Module): Loss function for labeled data (e.g., CombinedLoss).
        loss_fn_unlabeled (nn.Module, optional): Loss function for unlabeled data.
                                                 Defaults to BCEWithLogitsLoss if None.

    Returns:
        float: Average combined loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    if loss_fn_unlabeled is None:
        loss_fn_unlabeled = nn.BCEWithLogitsLoss()

    # Create an iterator for unlabeled data to cycle through it
    unlabeled_iter = iter(unlabeled_loader)

    for l_images, l_masks, l_depths, _ in labeled_loader:
        # Get labeled batch
        l_images = l_images.to(device)
        l_masks = l_masks.to(device)
        l_depths = l_depths.to(device)

        # Get unlabeled batch
        try:
            u_images, u_masks, u_depths, _ = next(unlabeled_iter)
        except StopIteration:
            # Restart iterator if exhausted
            unlabeled_iter = iter(unlabeled_loader)
            u_images, u_masks, u_depths, _ = next(unlabeled_iter)

        u_images = u_images.to(device)
        u_masks = u_masks.to(device)  # Soft pseudo-labels
        u_depths = u_depths.to(device)

        optimizer.zero_grad()

        # 1. Labeled Forward Pass
        l_logits = model(l_images, l_depths)
        l_loss = loss_fn_labeled(l_logits, l_masks)

        # 2. Unlabeled Forward Pass
        u_logits = model(u_images, u_depths)
        # Ensure soft targets are float and same shape
        if u_masks.ndim == 3:
            u_masks = u_masks.unsqueeze(1)
        u_loss = loss_fn_unlabeled(u_logits, u_masks.float())

        # Combined Loss
        # We weight them equally (1.0) as per standard consistency training
        loss = l_loss + u_loss

        loss.backward()
        optimizer.step()

        batch_size = l_images.size(0)  # Track based on labeled epochs
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation DataLoader.
        device (str): Device to run on.
        loss_fn (nn.Module): Loss function.

    Returns:
        tuple: (average_loss, mAP_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            loss = loss_fn(logits, masks)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store for mAP calculation
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate mAP over range 0.5 - 0.95
        map_score = calc_map_score(all_preds, all_targets)
    else:
        map_score = 0.0

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation mAP: {map_score}")

    return avg_loss, map_score


def predict(model, loader, device):
    """
    Runs inference on the test set using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test DataLoader.
        device (str): Device to run on.

    Returns:
        tuple: (ids, predictions)
            ids (list): List of image IDs.
            predictions (np.ndarray): Array of predicted probability masks (N, H, W).
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, _, depths, ids in loader:
            images = images.to(device)
            depths = depths.to(device)

            # 1. Original Prediction
            logits_orig = model(images, depths)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. TTA: Horizontal Flip
            images_flip = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (B, C, H, W)
            logits_flip = model(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)

            # Flip back result
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average predictions
            probs_avg = (probs_orig + probs_flip_back) / 2.0

            # Store results
            # Move to cpu and numpy
            probs_avg = probs_avg.cpu().numpy()

            # Remove channel dim if present (B, 1, H, W) -> (B, H, W)
            if probs_avg.ndim == 4:
                probs_avg = probs_avg.squeeze(1)

            all_preds.append(probs_avg)
            all_ids.extend(ids)

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = np.array([])

    return all_ids, all_preds


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=7, mode="max", delta=0.0, save_path="checkpoint.pth"):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            mode (str): 'min' for loss, 'max' for metric like mAP.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.val_score_max = -np.Inf

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        else:
            if self.mode == "min":
                if score < self.best_score - self.delta:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0
                else:
                    self.counter += 1
            elif self.mode == "max":
                if score > self.best_score + self.delta:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0
                else:
                    self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        torch.save(model.state_dict(), self.save_path)
