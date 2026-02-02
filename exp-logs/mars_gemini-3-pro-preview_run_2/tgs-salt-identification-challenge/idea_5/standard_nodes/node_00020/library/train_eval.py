import torch
import numpy as np
from library.utils import calc_iou_metric


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (str or torch.device): Device to compute on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Unpack batch: image, mask, depth, id
        images, masks, depths, _ = batch

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass: Model expects (x, depth)
        outputs = model(images, depths)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set and finds the optimal binarization threshold.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): The loss function.
        device (str or torch.device): Device to compute on.

    Returns:
        tuple: (average_loss, best_map_score, best_threshold)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch: image, mask, depth, id
            images, masks, depths, _ = batch

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            batch_size = images.size(0)
            dataset_size += batch_size

            outputs = model(images, depths)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to get probabilities [0, 1]
            preds = torch.sigmoid(outputs)

            # Store predictions and targets for global metric calculation
            # Move to CPU to save GPU memory during accumulation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Linear search for optimal threshold
    # We sweep from 0.3 to 0.75 to find the threshold that maximizes mAP
    thresholds = np.arange(0.3, 0.76, 0.05)
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        # Calculate mAP at this binarization threshold
        score = calc_iou_metric(all_preds, all_targets, binarization_threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(
        f"Validation Results - Loss: {val_loss}, Best mAP: {best_score}, Best Threshold: {best_threshold}"
    )

    return val_loss, best_score, best_threshold


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Data loader (test or val).
        device (str or torch.device): Device to compute on.

    Returns:
        tuple: (predictions_array, ids_list)
               predictions_array is (N, 1, H, W) float probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle variable unpacking depending on whether masks are present
            if len(batch) == 3:
                images, depths, ids = batch
            else:
                images, _, depths, ids = batch

            images = images.to(device)
            depths = depths.to(device)

            # 1. Original Prediction
            out_orig = model(images, depths)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Flipped Prediction (Horizontal Flip)
            # Flip width dimension (dim 3 for B,C,H,W)
            images_flipped = torch.flip(images, dims=[3])
            out_flipped = model(images_flipped, depths)
            prob_flipped = torch.sigmoid(out_flipped)

            # Flip predictions back to original orientation
            prob_flipped_back = torch.flip(prob_flipped, dims=[3])

            # Average the predictions
            prob_avg = (prob_orig + prob_flipped_back) / 2.0

            all_preds.append(prob_avg.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=10, min_delta=0, mode="max"):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will stop
                        when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def __call__(self, score):
        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
