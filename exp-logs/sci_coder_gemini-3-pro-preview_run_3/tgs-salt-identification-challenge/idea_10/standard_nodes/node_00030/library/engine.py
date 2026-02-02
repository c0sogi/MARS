import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, calc_map, rle_encode


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """
    Trains the model for one epoch using Automatic Mixed Precision.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (e.g., DeepSupervisionLoss).
        optimizer (Optimizer): Optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Device to train on ('cuda' or 'cpu').
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Automatic Mixed Precision Context
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch}: Train Loss = {losses.avg:.6f}")
    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates Loss and Mean Average Precision (mAP).

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, map_score)
    """
    model.eval()
    losses = AverageMeter()

    # Lists to store full-dataset predictions for mAP calculation
    all_preds = []
    all_targets = []

    # Calculate padding offsets for unpadding (Center Crop)
    # Config.IMG_SIZE is 128, Config.ORIG_SIZE is 101
    diff = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_top = diff // 2
    pad_left = diff // 2

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)

            losses.update(loss.item(), images.size(0))

            # Handle Deep Supervision output (list of tensors)
            # We take the last output (finest scale) for metric calculation
            if isinstance(outputs, (list, tuple)):
                final_logits = outputs[-1]
            else:
                final_logits = outputs

            # Apply Sigmoid to get probabilities
            preds = torch.sigmoid(final_logits)

            # Unpad: Crop the center 101x101 region
            # Tensor shape: (B, 1, H, W)
            preds_cropped = preds[
                :,
                :,
                pad_top : pad_top + Config.ORIG_SIZE,
                pad_left : pad_left + Config.ORIG_SIZE,
            ]
            masks_cropped = masks[
                :,
                :,
                pad_top : pad_top + Config.ORIG_SIZE,
                pad_left : pad_left + Config.ORIG_SIZE,
            ]

            # Move to CPU and store
            all_preds.append(preds_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())

    map_score = 0.0
    if len(all_preds) > 0:
        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Squeeze channel dimension: (N, 1, H, W) -> (N, H, W)
        all_preds = all_preds.squeeze(1)
        all_targets = all_targets.squeeze(1)

        # Calculate mAP over range of thresholds
        map_score = calc_map(all_preds, all_targets, threshold=0.5)

    print(f"Validation: Loss = {losses.avg:.6f}, mAP = {map_score:.6f}")
    return losses.avg, map_score


def predict_tta(model, images):
    """
    Performs Test-Time Augmentation (Horizontal Flip) on a batch of images.

    Args:
        model (nn.Module): The model.
        images (torch.Tensor): Input images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability maps (B, 1, H, W).
    """
    model.eval()
    with torch.no_grad():
        # 1. Forward pass original
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            logits = outputs[-1]
        else:
            logits = outputs
        prob = torch.sigmoid(logits)

        # 2. Forward pass flipped (Horizontal flip on width dim=3)
        images_flip = torch.flip(images, dims=[3])
        outputs_flip = model(images_flip)
        if isinstance(outputs_flip, (list, tuple)):
            logits_flip = outputs_flip[-1]
        else:
            logits_flip = outputs_flip
        prob_flip = torch.sigmoid(logits_flip)

        # 3. Flip back
        prob_flip_back = torch.flip(prob_flip, dims=[3])

        # 4. Average
        avg_prob = (prob + prob_flip_back) / 2.0

    return avg_prob


def generate_submission(
    model, loader, device, output_path=Config.SUBMISSION_PATH, threshold=0.5
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Applies TTA, unpadding, and RLE encoding.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Test data loader.
        device (str): Device to run inference on.
        output_path (str): Path to save the submission CSV.
        threshold (float): Threshold to binarize probabilities.
    """
    print("Generating submission...")
    model.eval()

    submission_ids = []
    submission_rles = []

    # Calculate padding offsets
    diff = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_top = diff // 2
    pad_left = diff // 2

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device, dtype=torch.float32)

            # Predict with TTA
            # Note: We don't use AMP here to ensure maximum precision for inference,
            # though it wouldn't hurt.
            probs = predict_tta(model, images)

            # Unpad
            probs_cropped = probs[
                :,
                :,
                pad_top : pad_top + Config.ORIG_SIZE,
                pad_left : pad_left + Config.ORIG_SIZE,
            ]

            # Convert to numpy
            probs_np = probs_cropped.cpu().numpy().squeeze(1)  # (B, H, W)

            # Process batch
            for i in range(len(ids)):
                img_id = ids[i]
                prob_map = probs_np[i]

                # Threshold
                binary_mask = (prob_map > threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(binary_mask)

                submission_ids.append(img_id)
                submission_rles.append(rle)

    # Create DataFrame
    df = pd.DataFrame({"id": submission_ids, "rle_mask": submission_rles})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
