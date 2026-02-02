import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, List
from library.config import Config
from library.utils import calculate_metric


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch using Gradient Accumulation.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        data_loader: Training data loader.
        device: Computation device (CPU/GPU).
        criterion: Loss function.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Gradient accumulation settings
    accumulation_steps = Config.gradient_accumulation_steps

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer every 'accumulation_steps' batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate statistics (multiply by accumulation_steps to get back original scale for logging)
        running_loss += loss.item() * accumulation_steps * batch_size
        dataset_size += batch_size

    # Handle any remaining gradients if dataset size is not divisible by accumulation steps
    if len(data_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def validate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        data_loader: Validation data loader.
        device: Computation device.
        criterion: Loss function.

    Returns:
        Tuple[float, float]: Average Loss and Mean F1-Score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate F1 Score
    epoch_f1 = calculate_metric(all_targets, all_preds, threshold=0.5)

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation F1-Score: {epoch_f1}")

    return epoch_loss, epoch_f1


def predict(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = Config.use_tta,
) -> np.ndarray:
    """
    Performs inference on the data loader. Supports Test Time Augmentation (TTA).

    Args:
        model: The PyTorch model.
        data_loader: Inference data loader.
        device: Computation device.
        use_tta: Whether to use horizontal flip TTA.

    Returns:
        np.ndarray: Predicted probabilities of shape (N, num_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            # Forward pass original
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if use_tta:
                # Forward pass flipped (Horizontal Flip)
                # dims=[3] corresponds to width in (B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def generate_submission(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    test_df: pd.DataFrame,
    device: torch.device,
    output_path: str = Config.submission_path,
) -> None:
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model: The trained PyTorch model.
        data_loader: Test data loader.
        test_df: DataFrame containing test image IDs.
        device: Computation device.
        output_path: Path to save the submission file.
    """
    print("Generating submission...")

    # Get probabilities
    probs = predict(model, data_loader, device, use_tta=Config.use_tta)

    # Convert probabilities to labels
    # Threshold = 0.5
    predictions = []
    class_labels = Config.class_labels

    for i in range(len(probs)):
        row_probs = probs[i]
        # Get indices where probability > 0.5
        indices = np.where(row_probs > 0.5)[0]

        if len(indices) > 0:
            labels = [class_labels[idx] for idx in indices]
            label_str = " ".join(labels)
        else:
            # Fallback if no class exceeds threshold
            # Based on dataset analysis, 'healthy' is a valid class.
            # We can either pick the max or default to healthy.
            # Given the metric is F1, picking max is safer than empty.
            max_idx = np.argmax(row_probs)
            label_str = class_labels[max_idx]

        predictions.append(label_str)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"image": test_df["image"], "labels": predictions})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
