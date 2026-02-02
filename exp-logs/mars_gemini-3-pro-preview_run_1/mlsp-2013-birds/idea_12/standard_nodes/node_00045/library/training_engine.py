import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utilities import AverageMeter, calculate_roc_auc


def train_one_epoch(model, loader, optimizer, device, config, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Torch device.
        config: Configuration object.
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    # Ensure numpy is seeded via global seed from utilities, but for safety in loop:
    # We rely on the global seed set in main.

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Mixup
        if config.MIXUP_ALPHA > 0:
            # Cite solution_lesson_node_00043: Use framework-native random functions for determinism
            lam = (
                torch.distributions.beta.Beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                .sample()
                .item()
            )
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index, :]
            # Mix labels (multi-label compatible)
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def validate(model, loader, device, config):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: Torch device.
        config: Configuration object.

    Returns:
        dict: Dictionary containing 'loss' and 'score' (ROC AUC).
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        # Correct loop structure
        for batch_idx, (images, labels, _) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for metric calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation Loss: {losses.avg:.6f}, ROC AUC: {score:.6f}")

    return {"loss": losses.avg, "score": score}


def update_swa(swa_model, model):
    """
    Updates the SWA model parameters with the current model's parameters.

    Args:
        swa_model: The AveragedModel instance.
        model: The current training model.
    """
    swa_model.update_parameters(model)


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model (or SWA model).
        loader: DataLoader (Validation or Test).
        device: Torch device.

    Returns:
        np.ndarray: Averaged probability predictions (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)

            # 1. Standard Forward Pass
            logits_std = model(images)
            probs_std = torch.sigmoid(logits_std)

            # 2. Flipped Forward Pass (Horizontal Flip)
            # Assuming images are (B, C, H, W), flip last dimension
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # Average Probabilities
            probs_avg = (probs_std + probs_flip) / 2.0

            all_preds.append(probs_avg.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def save_submission(predictions, test_metadata_path, output_path):
    """
    Formats and saves the submission file.

    Args:
        predictions (np.ndarray): Probability matrix (N_samples, 19).
        test_metadata_path (str): Path to test.csv to retrieve rec_ids.
        output_path (str): Path to save submission.csv.
    """
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

    df_test = pd.read_csv(test_metadata_path)
    rec_ids = df_test["rec_id"].values

    if len(rec_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(rec_ids)} test IDs vs {len(predictions)} predictions."
        )

    submission_rows = []
    num_classes = predictions.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_id in range(num_classes):
            # Construct Id: rec_id * 100 + species_id
            row_id = rec_id * 100 + species_id
            prob = predictions[i, species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
