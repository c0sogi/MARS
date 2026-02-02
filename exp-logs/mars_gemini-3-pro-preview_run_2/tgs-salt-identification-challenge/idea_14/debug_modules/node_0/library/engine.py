import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from library.utils import calculate_iou_map, rle_encode
from library.losses import CombinedLoss

# Constants for unpadding (128x128 -> 101x101)
# Albumentations PadIfNeeded with min_height=128, min_width=128 on 101x101 image
# usually pads 13 pixels on top/left and 14 pixels on bottom/right.
PAD_TOP = 13
PAD_LEFT = 13
ORIG_SIZE = 101


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Note: The 'Bernoulli Scalar Masking' logic (randomly zeroing out depth)
    is implemented within the SaltDataset class. This function receives the
    already processed depth tensors.
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
        # Model expects (x, depth)
        outputs = model(images, depths)

        # Calculate Loss (Combined BCE + Lovasz)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the mAP score calculated at multiple IoU thresholds.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, depths, _ in dataloader:
            batch_size = images.size(0)

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            outputs = model(images, depths)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # For intermediate validation monitoring, we use a default threshold
            # of 0.0 (logits) to binarize predictions for the mAP calculation.
            preds_binary = (outputs > 0).float()

            all_preds.append(preds_binary.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Squeeze channel dimension if present: (N, 1, H, W) -> (N, H, W)
    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
    if all_targets.ndim == 4:
        all_targets = all_targets.squeeze(1)

    # Calculate Mean Average Precision
    map_score = calculate_iou_map(all_targets, all_preds)

    return epoch_loss, map_score


def find_best_threshold(model, dataloader, device):
    """
    Performs a linear search over probability thresholds to find the one
    that maximizes the mAP on the validation set.
    """
    model.eval()
    all_logits = []
    all_targets = []

    # Collect all logits and targets
    with torch.no_grad():
        for images, masks, depths, _ in dataloader:
            images = images.to(device)
            depths = depths.to(device)
            masks = masks.to(device)

            outputs = model(images, depths)
            all_logits.append(outputs.cpu())
            all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()

    if all_targets.ndim == 4:
        all_targets = all_targets.squeeze(1)

    # Convert logits to probabilities
    all_probs = torch.sigmoid(all_logits).numpy()
    if all_probs.ndim == 4:
        all_probs = all_probs.squeeze(1)

    # Sweep thresholds
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_map = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (all_probs > t).astype(np.float32)
        score = calculate_iou_map(all_targets, preds)
        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Optimal Threshold Found: {best_thresh:.2f} with mAP: {best_map}")
    return best_thresh


def generate_submission(
    model, dataloader, threshold, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set.
    Features:
    - Forces depth to 0 (handling missing metadata).
    - Test-Time Augmentation (Horizontal Flip).
    - Unpadding (Center Crop).
    - RLE Encoding.
    """
    model.eval()
    ids_list = []
    rle_list = []

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        for images, _, _, ids in dataloader:
            images = images.to(device)

            # Force depth to 0 for test set ("Generalist" mode)
            zeros_depth = torch.zeros(
                (images.size(0), 1), device=device, dtype=torch.float32
            )

            # 1. Original Prediction
            out_orig = torch.sigmoid(model(images, zeros_depth))

            # 2. TTA: Horizontal Flip
            images_flip = torch.flip(images, [3])
            out_flip = torch.sigmoid(model(images_flip, zeros_depth))
            out_flip = torch.flip(out_flip, [3])

            # Average predictions
            avg_pred = (out_orig + out_flip) / 2.0

            # Center Crop to restore 101x101 resolution
            # Slicing based on Albumentations padding logic
            avg_pred = avg_pred[
                :, :, PAD_TOP : PAD_TOP + ORIG_SIZE, PAD_LEFT : PAD_LEFT + ORIG_SIZE
            ]

            # Binarize using optimal threshold
            pred_binary = (avg_pred > threshold).float().cpu().numpy()

            # Encode
            for i in range(len(ids)):
                # Extract single mask (H, W)
                mask = pred_binary[i, 0, :, :]
                rle = rle_encode(mask)
                ids_list.append(ids[i])
                rle_list.append(rle)

    # Save to CSV
    df = pd.DataFrame({"id": ids_list, "rle_mask": rle_list})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


class Trainer:
    """
    Orchestrates the training process, including Early Stopping, Checkpointing,
    and triggering submission generation.
    """

    def __init__(
        self, model, optimizer, criterion, device, patience=10, save_dir="./working"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        self.best_map = -1.0
        self.counter = 0

        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, "best_model.pth")

    def fit(self, train_loader, val_loader, test_loader=None, epochs=50):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )
            val_loss, val_map = validate(
                self.model, val_loader, self.criterion, self.device
            )

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val mAP: {val_map}"
            )

            # Checkpoint based on mAP
            if val_map > self.best_map:
                self.best_map = val_map
                self.counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.counter += 1

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                break

        # Load best model for inference
        print(
            f"Loading best model from {self.best_model_path} with mAP {self.best_map}"
        )
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

        # Post-Training: Find Threshold
        print("Finding optimal threshold on validation set...")
        best_thresh = find_best_threshold(self.model, val_loader, self.device)

        # Generate Submission if test loader is provided
        if test_loader is not None:
            print("Generating submission for test set...")
            generate_submission(self.model, test_loader, best_thresh, self.device)
