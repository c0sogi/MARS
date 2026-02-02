import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.dataset import mixup_data
from library.utils import mixup_criterion, calculate_roc_auc
from library.config import Config


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Saves the best model state.
    """

    def __init__(self, patience=5, mode="max", delta=0.0, save_path="checkpoint.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when the
                quantity monitored has stopped increasing.
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
        self.best_loss = np.inf

        if mode == "min":
            self.val_score_fn = lambda x: -x
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.val_score_fn(score) < self.val_score_fn(self.best_score) + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation metric decreases."""
        torch.save(model.state_dict(), self.save_path)


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch_idx, mixup_alpha
):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.
        epoch_idx: Current epoch index (for logging).
        mixup_alpha: Alpha parameter for Mixup.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, mixup_alpha, device
        )

        # Reshape targets for BCEWithLogitsLoss: (N,) -> (N, 1)
        targets_a = targets_a.view(-1, 1)
        targets_b = targets_b.view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reshape targets for BCEWithLogitsLoss: (N,) -> (N, 1)
            targets_reshaped = targets.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets_reshaped)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # For AUC, we need probabilities. Apply sigmoid.
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    epoch_loss = running_loss / dataset_size

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc_score = calculate_roc_auc(all_targets, all_preds)

    # Print metrics with full precision as requested
    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation ROC AUC: {auc_score}")

    return epoch_loss, auc_score


def predict(model, dataloader, device, use_tta=True):
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: Torch device.
        use_tta (bool): Whether to use 4-view Test Time Augmentation.

    Returns:
        np.array: Array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            if use_tta:
                # 4-view TTA: Original, H-Flip, V-Flip, Rotate 180 (H+V Flip)
                # Shape: (B, C, H, W)

                # Create augmented versions
                img_orig = inputs
                img_h = torch.flip(inputs, [3])  # Flip width
                img_v = torch.flip(inputs, [2])  # Flip height
                img_hv = torch.flip(inputs, [2, 3])  # Flip both

                # Stack along batch dimension for efficient processing
                # New shape: (4*B, C, H, W)
                batch_stack = torch.cat([img_orig, img_h, img_v, img_hv], dim=0)

                # Forward pass
                logits = model(batch_stack)
                probs = torch.sigmoid(logits)

                # Reshape back to (4, B, 1) to average
                # batch_stack was [B_orig, B_h, B_v, B_hv]
                batch_size = inputs.size(0)
                probs_reshaped = probs.view(4, batch_size, 1)

                # Average across the 4 views
                avg_probs = torch.mean(probs_reshaped, dim=0)
                all_preds.append(avg_probs.cpu())

            else:
                # Standard inference
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu())

    return torch.cat(all_preds).numpy().flatten()


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions and saves the submission CSV.

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: Torch device.
        output_path: Path to save the submission CSV.
    """
    print("Generating submission...")

    # Get predictions
    probs = predict(model, dataloader, device, use_tta=Config.USE_TTA)

    # Load test metadata to get IDs
    # We assume the dataloader iterates sequentially over the test set
    # consistent with the order in test_metadata.csv
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if len(probs) != len(test_meta_df):
        raise ValueError(
            f"Number of predictions ({len(probs)}) does not match number of test samples ({len(test_meta_df)})"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_meta_df["id"], "has_cactus": probs})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
