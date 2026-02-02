import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from library.config import Config
from library.utils import MetricMonitor, get_device
from library.losses import AsymmetricLoss, DistillationLoss


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    use_ema=False,
    ema_model=None,
):
    """
    Trains the model for one epoch.
    Handles both standard training (Teacher) and distillation (Student).
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Initialize losses
    # We initialize them here to ensure they are on the correct device if needed,
    # though these specific losses are stateless/functional mostly.
    criterion_asl = AsymmetricLoss().to(device)
    criterion_distill = DistillationLoss().to(device)

    for batch_idx, batch_data in enumerate(loader):
        # Determine if we are in distillation mode based on batch structure
        # Batch: (image, soft_label, hard_label) OR (image, hard_label)
        if len(batch_data) == 3:
            images, soft_targets, hard_targets = batch_data
            distillation = True
        else:
            images, hard_targets = batch_data
            soft_targets = None
            distillation = False

        images = images.to(device, non_blocking=True)
        hard_targets = hard_targets.to(device, non_blocking=True)
        if soft_targets is not None:
            soft_targets = soft_targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)

            if distillation:
                # DistillationLoss expects: student_logits, teacher_logits, hard_targets
                loss = criterion_distill(outputs, soft_targets, hard_targets)
            else:
                loss = criterion_asl(outputs, hard_targets)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if use_ema and ema_model is not None:
            ema_model.update(model)

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch {epoch} Train | {metric_monitor}")


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, logits, and targets.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = AsymmetricLoss().to(device)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            metric_monitor.update("Loss", loss.item())

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    print(f"Validation | {metric_monitor}")

    predictions = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    return metric_monitor.avg["Loss"], predictions, targets


def generate_soft_labels(models, loader, device, save_path, load_cached_data=True):
    """
    Generates soft labels (logits) for the training set using a teacher ensemble.
    Implements caching to avoid re-computation.

    Args:
        models (list): List of loaded PyTorch models (teachers).
        loader (DataLoader): DataLoader for the training set (deterministic transforms).
        device (torch.device): Computation device.
        save_path (str): Path to save/load the .npy file.
        load_cached_data (bool): Whether to attempt loading from cache.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(save_path):
        print(f"Loading cached soft labels from {save_path}")
        return np.load(save_path)

    print("Generating soft labels from teacher ensemble...")

    # Ensure models are in eval mode
    for model in models:
        model.eval()
        model.to(device)

    all_probs = []

    # 2. Compute Predictions
    # We iterate through the loader once, predicting with all models
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            batch_probs_sum = None

            # Ensemble averaging in probability space
            for model in models:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs

            # Average probabilities
            avg_probs = batch_probs_sum / len(models)
            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    final_probs = np.concatenate(all_probs, axis=0)

    # 3. Convert to Logits
    # DistillationLoss expects logits, so we invert the sigmoid.
    # Clip to avoid log(0) or log(1)
    eps = 1e-6
    final_probs = np.clip(final_probs, eps, 1 - eps)
    final_logits = np.log(final_probs / (1 - final_probs))

    # 4. Save to Cache
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_logits)
    print(f"Soft labels saved to {save_path}")

    return final_logits


def inference(model, loader, device, use_tta=True):
    """
    Runs inference on the test set.
    Supports Test-Time Augmentation (Horizontal Flip).

    Returns:
        ids (list): List of image IDs.
        probs (np.ndarray): Predicted probabilities.
    """
    model.eval()
    ids_list = []
    probs_list = []

    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(loader):
            images = images.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                # Pass 1: Original
                logits = model(images)
                probs = torch.sigmoid(logits)

                if use_tta:
                    # Pass 2: Horizontal Flip
                    # Flip along width dimension (dim 3 for N,C,H,W)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)

                    # Average
                    probs = (probs + probs_flipped) / 2.0

            probs_list.append(probs.cpu().numpy())
            ids_list.extend(ids)

    return ids_list, np.concatenate(probs_list, axis=0)


def find_best_threshold(targets, probs, step=0.05):
    """
    Finds the single global threshold that maximizes Micro F1 score.
    """
    best_thresh = 0.5
    best_score = 0.0

    # Search range [0.1, 0.9]
    thresholds = np.arange(0.1, 0.95, step)

    for thresh in thresholds:
        preds_bin = (probs > thresh).astype(int)
        score = f1_score(targets, preds_bin, average="micro")

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"Best Threshold: {best_thresh:.4f} | Best Micro F1: {best_score:.4f}")
    return best_thresh


def create_submission(ids, probs, threshold, save_path):
    """
    Generates the submission CSV file based on predictions and a threshold.
    """
    predictions = []

    for idx, img_id in enumerate(ids):
        # Get indices where probability > threshold
        pred_indices = np.where(probs[idx] > threshold)[0]

        # Format as space-separated string
        pred_str = " ".join(map(str, pred_indices))
        predictions.append(pred_str)

    df_sub = pd.DataFrame({"id": ids, "attribute_ids": predictions})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
