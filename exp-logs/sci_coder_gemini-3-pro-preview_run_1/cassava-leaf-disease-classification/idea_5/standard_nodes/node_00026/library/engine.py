import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that handles soft targets (probabilities) from MixUp/CutMix.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_classes) logits
        # target: (batch_size, num_classes) probabilities
        log_probs = F.log_softmax(x, dim=-1)
        loss = torch.sum(-target * log_probs, dim=-1)
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
    mixup_fn=None,
    scheduler=None,
) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    num_steps = 0

    # Define loss functions
    soft_criterion = SoftTargetCrossEntropy()
    hard_criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply MixUp / CutMix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        if mixup_fn is not None:
            # targets are (B, C) probabilities
            loss = soft_criterion(outputs, targets)
        else:
            # targets are (B,) class indices
            loss = hard_criterion(outputs, targets)

        # Normalize loss for gradient accumulation
        loss = loss / config.accum_iter
        loss.backward()

        # Update weights
        if (batch_idx + 1) % config.accum_iter == 0:
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # Step scheduler if it's per-iteration (optional, usually handled externally or per epoch)
            if scheduler is not None and hasattr(scheduler, "step_batch"):
                scheduler.step_batch()

        total_loss += loss.item() * config.accum_iter
        num_steps += 1

    avg_loss = total_loss / num_steps
    return avg_loss


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
) -> tuple:
    """
    Evaluates the model on the validation set.
    Returns (accuracy, average_loss).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples

    return accuracy, avg_loss


def inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
) -> list:
    """
    Performs inference on the test set using Test Time Augmentation (TTA).
    Returns a list of dictionaries containing 'image_id' and 'label'.
    """
    model.eval()
    predictions = []

    # Get image IDs from the dataset
    # We assume the loader preserves order and the dataset has a df attribute or similar
    # The CassavaDataset stores file_paths. We can extract IDs from there or the original dataframe.
    image_ids = loader.dataset.df["image_id"].values

    idx_counter = 0

    with torch.no_grad():
        for images in loader:
            # Handle case where loader returns (images, targets) or just images
            if isinstance(images, (list, tuple)):
                images = images[0]

            images = images.to(device)
            batch_size = images.size(0)

            # TTA Strategy: Original + Horizontal Flip + Vertical Flip
            # 1. Original
            out_orig = model(images)
            prob_orig = F.softmax(out_orig, dim=1)

            # 2. Horizontal Flip
            images_hflip = torch.flip(images, dims=[3])
            out_hflip = model(images_hflip)
            prob_hflip = F.softmax(out_hflip, dim=1)

            # 3. Vertical Flip
            images_vflip = torch.flip(images, dims=[2])
            out_vflip = model(images_vflip)
            prob_vflip = F.softmax(out_vflip, dim=1)

            # Average probabilities
            avg_probs = (prob_orig + prob_hflip + prob_vflip) / 3.0

            # Get predicted labels
            pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

            # Store results
            for i in range(batch_size):
                img_id = image_ids[idx_counter]
                label = pred_labels[i]
                predictions.append({"image_id": img_id, "label": label})
                idx_counter += 1

    return predictions


def generate_submission(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
):
    """
    Generates predictions and saves them to submission.csv.
    """
    print("Starting inference with TTA...")
    preds = inference(model, loader, device, config)

    df_sub = pd.DataFrame(preds)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    # Save submission
    df_sub.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
    print(df_sub.head())
