import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_metrics


def get_pos_weight(df):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class imbalance.
    pos_weight = (num_negatives / num_positives)

    Args:
        df (pd.DataFrame): DataFrame containing the training data with 'species_X' columns.

    Returns:
        torch.Tensor: Weights of shape (NumClasses,).
    """
    label_cols = [c for c in df.columns if c.startswith("species_")]
    labels = df[label_cols].values

    num_samples = len(labels)
    num_pos = labels.sum(axis=0)
    num_neg = num_samples - num_pos

    # Avoid division by zero
    num_pos = np.maximum(num_pos, 1)

    pos_weight = num_neg / num_pos
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, pos_weight=None, scheduler=None):
    """
    Trains the model for one epoch using Selective Signal-Noise Mixup.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training dataloader.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): 'cuda' or 'cpu'.
        pos_weight (torch.Tensor, optional): Class weights for BCE loss.
        scheduler (LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define Loss Function
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        batch_size = images.size(0)

        # Selective Signal-Noise Mixup
        # Strategy: Mix 'Signal' (labeled) samples with 'Noise' (unlabeled) samples.
        if Config.USE_SELECTIVE_MIXUP:
            # Identify Signal and Noise indices
            # Labels are binary vectors. Sum > 0 implies Signal.
            label_sums = labels.sum(dim=1)
            signal_indices = torch.where(label_sums > 0)[0]
            noise_indices = torch.where(label_sums == 0)[0]

            # Proceed only if both Signal and Noise samples exist in the current batch
            if len(signal_indices) > 0 and len(noise_indices) > 0:
                # For each signal sample, select a random noise sample from the batch
                # We use random choice with replacement for noise samples
                rand_noise_indices = noise_indices[
                    torch.randint(len(noise_indices), (len(signal_indices),))
                ]

                # Sample mixing coefficient lambda from Beta distribution
                # If Alpha=1.0, this is a Uniform(0,1) distribution
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

                # Apply Mixup to Images: Signal = lam * Signal + (1-lam) * Noise
                images[signal_indices] = (
                    lam * images[signal_indices]
                    + (1 - lam) * images[rand_noise_indices]
                )

                # Apply Mixup to Labels: Signal = lam * Signal + (1-lam) * 0
                # Since noise labels are all zeros, we simply scale the signal labels
                labels[signal_indices] = labels[signal_indices] * lam

                # Note: Pure noise samples (images[noise_indices]) are left unchanged
                # to allow the model to learn the background class distribution.

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, device, pos_weight=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader.
        device (str): 'cuda' or 'cpu'.
        pos_weight (torch.Tensor, optional): Class weights for BCE loss calculation.

    Returns:
        tuple: (average_loss, macro_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        auc_score = calculate_metrics(all_targets, all_preds)
    else:
        auc_score = 0.0

    return epoch_loss, auc_score


def inference_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from 4 variants: Original + 3 Time-Roll shifts.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test dataloader.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (rec_ids, probabilities)
            rec_ids: numpy array of recording IDs.
            probabilities: numpy array of shape (N, NumClasses).
    """
    model.eval()
    results = []
    ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            rec_ids = batch["rec_id"]

            B, C, H, W = images.shape

            # Define TTA shifts (Time axis is dim 3)
            # Variants: Original (0), 25%, 50%, 75% shifts
            shifts = [0, W // 4, W // 2, (3 * W) // 4]

            batch_probs = []

            for shift in shifts:
                if shift == 0:
                    inputs = images
                else:
                    # Circular shift along the time axis
                    inputs = torch.roll(images, shifts=shift, dims=3)

                logits = model(inputs)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs)

            # Average predictions across TTA variants
            # Shape: (4, B, NumClasses) -> (B, NumClasses)
            avg_probs = torch.stack(batch_probs).mean(dim=0)

            results.append(avg_probs.cpu().numpy())
            ids.append(rec_ids.numpy())

    if len(ids) > 0:
        return np.concatenate(ids), np.concatenate(results)
    else:
        return np.array([]), np.array([])


def save_submission(rec_ids, probs, output_path):
    """
    Formats and saves the submission file.
    Format: Id,Probability
    Id = rec_id * 100 + species_id

    Args:
        rec_ids (np.array): Array of recording IDs.
        probs (np.array): Array of probabilities (N, NumClasses).
        output_path (str): Path to save the CSV.
    """
    submission_rows = []
    num_classes = probs.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_id in range(num_classes):
            # Construct unique Id
            row_id = int(rec_id * 100 + species_id)
            probability = probs[i, species_id]
            submission_rows.append({"Id": row_id, "Probability": probability})

    df_sub = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
