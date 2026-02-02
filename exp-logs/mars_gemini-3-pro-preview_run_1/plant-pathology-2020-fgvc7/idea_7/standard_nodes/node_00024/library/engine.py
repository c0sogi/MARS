import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_roc_auc


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: The DataLoader for the training set.
        optimizer: The optimizer used for updating weights.
        criterion: The loss function.
        device: The device (CPU/GPU) to perform computation on.
        epoch: The current epoch number (for logging).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")

    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: The DataLoader for the validation set.
        criterion: The loss function.
        device: The device (CPU/GPU) to perform computation on.

    Returns:
        tuple: (average_loss, auc_score, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Softmax to get probabilities for ROC AUC calculation
            # The model outputs logits, but AUC requires probabilities.
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Mean Column-wise ROC AUC
    auc_score = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation AUC: {auc_score}")

    return epoch_loss, auc_score, all_preds, all_targets


def predict_and_save(model, dataloader, device, save_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Implements Test-Time Augmentation (TTA) by averaging predictions of
    original and horizontally flipped images.

    Args:
        model: The PyTorch model to use for inference.
        dataloader: The DataLoader for the test set (must return 'image_id').
        device: The device (CPU/GPU) to perform computation on.
        save_path: The file path to save the submission CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Check if TTA is enabled in Config
    use_tta = getattr(Config, "USE_TTA", False)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)

            if "image_id" in batch:
                all_ids.extend(batch["image_id"])

            # 1. Forward pass with original images
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # 2. Forward pass with horizontally flipped images
                # Flip along width (dim 3 for B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # 3. Average predictions
                probs = (probs + probs_flipped) / 2.0

            all_preds.append(probs.cpu().numpy())

    final_preds = np.concatenate(all_preds, axis=0)

    # Create submission DataFrame
    # Columns must match the order of Config.TARGET_COLS used during training
    df = pd.DataFrame(final_preds, columns=Config.TARGET_COLS)

    # Insert image_id column
    if all_ids:
        df.insert(0, "image_id", all_ids)
    else:
        print("Warning: No image_ids found in dataloader outputs.")

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
