import torch
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import calculate_map, rle_encode
from library.losses import Phase1Loss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    # Consistent Phase1Loss (BCE + Dice) throughout
    criterion = Phase1Loss().to(device)

    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for batch in dataloader:
        images, masks, _ = batch
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass with Deep Supervision
        # Model returns (logits, aux2, aux3) in training mode
        logits, aux2, aux3 = model(images)

        # Calculate losses
        loss_main = criterion(logits, masks)
        loss_aux2 = criterion(aux2, masks)
        loss_aux3 = criterion(aux3, masks)

        # Weighted sum for deep supervision
        loss = loss_main + 0.5 * loss_aux2 + 0.5 * loss_aux3

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device, epoch):
    """
    Evaluates the model on the validation set.
    Calculates Loss and Mean Average Precision (mAP).
    """
    model.eval()
    criterion = Phase1Loss().to(device)

    running_loss = 0.0
    map_scores = []
    dataset_size = len(dataloader.dataset)

    # Padding indices for un-padding (128x128 -> 101x101)
    # Pad H: (13, 14), Pad W: (13, 14)
    start_idx = 13
    end_idx = 128 - 14  # 114

    with torch.no_grad():
        for batch in dataloader:
            images, masks, _ = batch
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass (returns only logits in eval mode)
            logits = model(images)

            # Calculate validation loss
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate mAP
            # 1. Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # 2. Un-pad predictions and targets to original 101x101 size
            # Slicing: [start:end]
            if probs.dim() == 4:
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]
            else:
                probs_cropped = probs[:, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, start_idx:end_idx, start_idx:end_idx]

            # 3. Calculate mAP for the batch
            # calculate_map expects probabilities and handles thresholding internally
            batch_map = calculate_map(probs_cropped, masks_cropped)
            map_scores.append(batch_map)

    avg_loss = running_loss / dataset_size
    avg_map = np.mean(map_scores)

    return avg_loss, avg_map


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    Applies Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()

    predictions = []
    ids_list = []

    # Padding indices for un-padding
    start_idx = 13
    end_idx = 114

    print(f"Generating submission to {output_path}...")

    with torch.no_grad():
        for batch in dataloader:
            images, ids = batch
            images = images.to(device)

            # 1. Forward Pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. Forward Pass (Horizontal Flip TTA)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            # Flip back predictions
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            # 3. Average Predictions
            avg_probs = (probs + probs_flipped_back) / 2.0

            # 4. Un-pad to original size (101x101)
            if avg_probs.dim() == 4:
                avg_probs = avg_probs[:, :, start_idx:end_idx, start_idx:end_idx]
                # Remove channel dim: (B, 1, H, W) -> (B, H, W)
                avg_probs = avg_probs.squeeze(1)

            # Convert to numpy
            avg_probs_np = avg_probs.cpu().numpy()

            # 5. Process Batch
            for i in range(len(ids)):
                img_id = ids[i]
                prob_map = avg_probs_np[i]

                # Threshold at 0.5
                binary_mask = (prob_map > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(binary_mask)

                ids_list.append(img_id)
                predictions.append(rle)

    # Create DataFrame
    df = pd.DataFrame({"id": ids_list, "rle_mask": predictions})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
