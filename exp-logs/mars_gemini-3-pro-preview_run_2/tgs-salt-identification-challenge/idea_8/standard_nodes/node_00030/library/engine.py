import torch
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from library.config import DEVICE, ORIG_SIZE, IMG_SIZE, SUBMISSION_PATH
from library.utils import rle_encode, metric_map


def crop_to_original(tensor, orig_size=ORIG_SIZE):
    """
    Center crops the tensor to the original size (101x101).
    Assumes tensor is (B, C, H, W) or (B, H, W).
    """
    if tensor.ndim == 4:
        h, w = tensor.shape[2], tensor.shape[3]
        start_h = (h - orig_size) // 2
        start_w = (w - orig_size) // 2
        return tensor[
            :, :, start_h : start_h + orig_size, start_w : start_w + orig_size
        ]
    elif tensor.ndim == 3:
        h, w = tensor.shape[1], tensor.shape[2]
        start_h = (h - orig_size) // 2
        start_w = (w - orig_size) // 2
        return tensor[:, start_h : start_h + orig_size, start_w : start_w + orig_size]
    else:
        raise ValueError(f"Unsupported tensor shape for cropping: {tensor.shape}")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device=DEVICE):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks, depths, _ in dataloader:
        batch_size = images.size(0)

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, depths)

        # Calculate loss
        loss = loss_fn(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, loss_fn, device=DEVICE):
    """
    Evaluates the model on the validation set and performs linear search
    for the optimal binarization threshold.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, _ in dataloader:
            batch_size = images.size(0)

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Forward pass
            logits = model(images, depths)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Crop back to original 101x101 for accurate metric calculation
            probs_cropped = crop_to_original(probs)
            masks_cropped = crop_to_original(masks)

            # Store on CPU
            all_probs.append(probs_cropped.cpu().numpy())
            all_masks.append(masks_cropped.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Squeeze channel dim if exists: (N, 1, H, W) -> (N, H, W)
    if all_probs.ndim == 4:
        all_probs = all_probs.squeeze(1)
    if all_masks.ndim == 4:
        all_masks = all_masks.squeeze(1)

    # Linear Search for Optimal Threshold
    # We search for the probability threshold that maximizes the competition metric
    thresholds = np.linspace(0.3, 0.7, 21)  # 0.30, 0.32, ..., 0.70
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        # Binarize predictions based on current threshold
        # metric_map uses > 0.5 internally, so we pass 0s and 1s
        binary_preds = (all_probs > t).astype(np.uint8)

        # Calculate metric
        score = metric_map(binary_preds, all_masks)

        if score > best_score:
            best_score = score
            best_threshold = t

    return val_loss, best_score, best_threshold


def predict_test(model, dataloader, best_threshold, device=DEVICE):
    """
    Generates predictions for the test set using TTA and the optimal threshold.
    Saves the result to submission.csv.
    """
    model.eval()
    ids_list = []
    rle_list = []

    print(f"Generating predictions with threshold: {best_threshold:.4f}...")

    with torch.no_grad():
        for images, _, depths, ids in dataloader:
            images = images.to(device)

            # Depth handling:
            # Dataloader provides 0.0 for test set (mean of standardized training depths).
            # This aligns with the Depth Dropout strategy.
            depths = depths.to(device)

            # --- Test Time Augmentation (TTA) ---

            # 1. Original Prediction
            logits_orig = model(images, depths)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip Prediction
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped, depths)
            probs_flipped = torch.sigmoid(logits_flipped)
            # Flip back
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            # Average
            probs_avg = (probs_orig + probs_flipped_back) / 2.0

            # --- Post Processing ---

            # Crop to original size (101x101)
            probs_cropped = crop_to_original(probs_avg)

            # Convert to numpy
            probs_np = probs_cropped.cpu().numpy()

            # Squeeze channel dimension: (B, 1, H, W) -> (B, H, W)
            if probs_np.ndim == 4:
                probs_np = probs_np.squeeze(1)

            # Binarize and Encode
            for i in range(probs_np.shape[0]):
                pred_mask = (probs_np[i] > best_threshold).astype(np.uint8)
                rle = rle_encode(pred_mask)

                ids_list.append(ids[i])
                rle_list.append(rle)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": ids_list, "rle_mask": rle_list})

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
