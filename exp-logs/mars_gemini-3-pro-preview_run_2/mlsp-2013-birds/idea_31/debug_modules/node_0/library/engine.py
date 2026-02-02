import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import compute_auc
from library.data import BirdDataset, get_transforms


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    Supports both standard training and distillation if 'soft_target' is present in batch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        # Check for soft targets (for distillation phase)
        soft_targets = None
        if "soft_target" in batch:
            soft_targets = batch["soft_target"].to(device)

        optimizer.zero_grad()

        logits = model(images)

        # The DistillationLoss criterion handles the logic:
        # If soft_targets is None, it computes standard BCE.
        # If soft_targets is provided, it computes Weighted Distillation loss.
        loss = criterion(logits, targets, soft_targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(images)

            # Validation is always computed against hard ground truth labels
            loss = criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute AUC
    auc_score = compute_auc(all_targets, all_preds)

    return avg_loss, auc_score


def predict_tta(models, images, device):
    """
    Performs inference using Cyclic Test-Time Augmentation (TTA).

    Args:
        models (list or nn.Module): Single model or list of models (ensemble).
        images (np.ndarray): Raw images array (N, H, W, C).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Averaged probabilities (N, Num_Classes).
    """
    if not isinstance(models, list):
        models = [models]

    for model in models:
        model.eval()

    num_samples = len(images)
    num_classes = Config.NUM_CLASSES

    # Accumulator for averaged probabilities across TTA steps
    final_probs = np.zeros((num_samples, num_classes), dtype=np.float32)

    with torch.no_grad():
        # Iterate through defined TTA steps (e.g., 4 steps -> 0.0, 0.25, 0.5, 0.75 shift)
        for step in range(Config.TTA_STEPS):
            shift = step / float(Config.TTA_STEPS)

            # Get transforms with the specific deterministic shift
            transforms = get_transforms(phase="test", tta_shift=shift)

            # Create a temporary dataset and loader for this TTA view
            dataset = BirdDataset(images, transforms=transforms)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            step_preds_list = []

            for batch in loader:
                imgs = batch["image"].to(device)

                # Accumulate probabilities from all models for this batch
                batch_probs_sum = torch.zeros(
                    (imgs.size(0), num_classes), device=device
                )

                for model in models:
                    logits = model(imgs)
                    batch_probs_sum += torch.sigmoid(logits)

                # Average across models
                batch_avg_probs = batch_probs_sum / len(models)
                step_preds_list.append(batch_avg_probs.cpu().numpy())

            # Concatenate predictions for this TTA step
            step_probs = np.concatenate(step_preds_list, axis=0)

            # Add to final accumulator
            final_probs += step_probs

    # Average across TTA steps
    final_probs /= Config.TTA_STEPS

    return final_probs


def save_submission(rec_ids, probs, output_path):
    """
    Formats and saves the submission file.

    Mapping Logic:
    Id = rec_id * 100 + species_id

    Args:
        rec_ids (np.ndarray): Array of recording IDs.
        probs (np.ndarray): Predicted probabilities (N_samples, N_classes).
        output_path (str): Path to save the CSV.
    """
    ids = []
    probabilities = []

    num_classes = probs.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_id in range(num_classes):
            # Construct the unique Id required by the competition format
            unique_id = int(rec_id * 100 + species_id)
            prob = probs[i, species_id]

            ids.append(unique_id)
            probabilities.append(prob)

    df = pd.DataFrame({"Id": ids, "Probability": probabilities})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
