import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import calculate_multilabel_auc
from library.dataset import Mixup


def get_weighted_loss(device):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class frequency
    in the training set to handle class imbalance.

    pos_weight[i] = number_of_negatives[i] / number_of_positives[i]
    """
    df = pd.read_csv(Config.TRAIN_CSV)
    label_cols = [c for c in df.columns if c.startswith("species_")]

    # Calculate counts
    pos_counts = df[label_cols].sum().values
    total_counts = len(df)
    neg_counts = total_counts - pos_counts

    # Avoid division by zero for safe stability
    pos_counts = np.maximum(pos_counts, 1)

    # Calculate weights
    pos_weights = neg_counts / pos_counts
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    return criterion


def mixup_criterion(criterion, preds, targets_a, targets_b, lam):
    """
    Calculates the mixup loss: lambda * loss(a) + (1 - lambda) * loss(b)
    """
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using SAM optimizer and Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA)

    # Progress bar for monitoring
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False, disable=True)

    for batch in dataloader:
        images, labels, ids = batch
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        # Mixup returns: mixed_images, labels_a, labels_b, lam
        mixed_images, labels_a, labels_b, lam = mixup_fn((images, labels, ids))

        # --- SAM Step 1 ---
        # Forward pass
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # Backward pass
        loss.backward()

        # First step: climb to local maximum (perturb weights)
        optimizer.first_step(zero_grad=True)

        # --- SAM Step 2 ---
        # Forward pass again at perturbed state
        outputs_adv = model(mixed_images)
        loss_adv = mixup_criterion(criterion, outputs_adv, labels_a, labels_b, lam)

        # Backward pass
        loss_adv.backward()

        # Second step: update weights to flat minimum
        optimizer.second_step(zero_grad=True)

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        pbar.update(1)

    pbar.close()
    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro-AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, ids = batch
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    auc_score = calculate_multilabel_auc(all_targets, all_preds)

    return epoch_loss, auc_score


def predict_cyclic_tta(model, dataloader, device):
    """
    Performs inference using Cyclic Test-Time Augmentation (TTA).
    Averages predictions across 4 cyclic shifts: [0, 25%, 50%, 75%] of width.
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Shifts as fractions of width (Time axis)
    shifts_ratios = [0.0, 0.25, 0.50, 0.75]

    with torch.no_grad():
        for batch in dataloader:
            images, labels, ids = batch
            images = images.to(device)

            # images shape: (B, C, H, W)
            # W is dimension 3
            width = images.shape[3]

            batch_probs = []

            for ratio in shifts_ratios:
                shift_pixels = int(width * ratio)

                # Cyclic roll along time axis
                if shift_pixels > 0:
                    shifted_images = torch.roll(images, shifts=shift_pixels, dims=3)
                else:
                    shifted_images = images

                outputs = model(shifted_images)
                probs = torch.sigmoid(outputs)
                batch_probs.append(probs)

            # Stack and average across TTA steps
            # Shape: (TTA_Steps, Batch, Num_Classes) -> Mean -> (Batch, Num_Classes)
            batch_probs = torch.stack(batch_probs)
            avg_probs = torch.mean(batch_probs, dim=0)

            all_preds.append(avg_probs.cpu().numpy())
            all_ids.append(ids.numpy())

    final_preds = np.concatenate(all_preds, axis=0)
    final_ids = np.concatenate(all_ids, axis=0)

    return final_ids, final_preds
