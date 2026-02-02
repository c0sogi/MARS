import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import FRUNet
from library.utils import calculate_fbeta, optimize_threshold, rle_encode


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def weighted_bce_loss(preds, targets, masks, pos_weight, epsilon=1e-7):
    """
    Computes weighted binary cross entropy on probabilities, masked by validity mask.

    Args:
        preds (torch.Tensor): Model outputs (probabilities) of shape (B, 1, H, W).
        targets (torch.Tensor): Ground truth labels of shape (B, 1, H, W).
        masks (torch.Tensor): Validity masks of shape (B, 1, H, W).
        pos_weight (float): Weight for the positive class.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Clamp predictions to avoid log(0)
    preds = torch.clamp(preds, epsilon, 1.0 - epsilon)

    # Weighted BCE formula: - [ pos_weight * y * log(p) + (1-y) * log(1-p) ]
    loss_map = -(
        targets * torch.log(preds) * pos_weight + (1 - targets) * torch.log(1 - preds)
    )

    # Apply validity mask (only calculate loss on valid pixels)
    masked_loss = loss_map * masks

    # Normalize by the number of valid pixels
    loss = masked_loss.sum() / (masks.sum() + epsilon)

    return loss


def train_one_epoch(model, dataloader, optimizer, device, pos_weight):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0

    for volumes, labels, masks, _ in dataloader:
        volumes = volumes.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(volumes)

        loss = weighted_bce_loss(outputs, labels, masks, pos_weight)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, device, pos_weight):
    """
    Validates the model and finds the optimal threshold for F0.5 score.
    Returns average loss, best F0.5 score, and the corresponding threshold.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for volumes, labels, masks, _ in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(volumes)

            # Calculate validation loss
            loss = weighted_bce_loss(outputs, labels, masks, pos_weight)
            running_loss += loss.item()

            # Collect valid predictions for metric calculation
            # Flatten and filter by mask to save memory and ensure accuracy
            valid_mask_np = masks.cpu().numpy().astype(bool)
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(outputs_np[valid_mask_np])
            all_labels.append(labels_np[valid_mask_np])

    avg_loss = running_loss / len(dataloader)

    # Concatenate all valid pixels from the epoch
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Find the threshold that maximizes F0.5 on this validation set
        best_thresh, best_f05 = optimize_threshold(all_preds, all_labels, beta=0.5)
    else:
        best_thresh = 0.5
        best_f05 = 0.0

    return avg_loss, best_f05, best_thresh


def generate_submission(model, device, threshold):
    """
    Generates predictions for the test set, reconstructs full fragments,
    performs RLE encoding, and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Load test dataset
    test_dataset = InkDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Dictionary to store patch predictions: fragment_id -> list of (x, y, w, h, pred_mask)
    fragment_preds = {}

    with torch.no_grad():
        for volumes, _, masks, sample_ids in test_loader:
            volumes = volumes.to(device)
            outputs = model(volumes)

            # Binarize using the optimized threshold
            preds = (outputs >= threshold).float().cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                # Parse sample_id: {fragment_id}_{y}_{x}
                parts = sample_id.split("_")
                fid = parts[0]
                y = int(parts[1])
                x = int(parts[2])

                # Get dimensions from prediction shape
                pred_h, pred_w = preds[i, 0].shape

                if fid not in fragment_preds:
                    fragment_preds[fid] = []

                fragment_preds[fid].append((x, y, pred_w, pred_h, preds[i, 0]))

    # Reconstruct fragments and encode
    submission_rows = []

    for fid, patches in fragment_preds.items():
        # Determine full fragment size by reading the mask file
        mask_path = os.path.join(Config.INPUT_DIR, "test", fid, "mask.png")
        if os.path.exists(mask_path):
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
        else:
            # Fallback: infer from patch coordinates
            max_x = max(p[0] + p[2] for p in patches)
            max_y = max(p[1] + p[3] for p in patches)
            W, H = max_x, max_y

        # Create canvas
        full_mask = np.zeros((H, W), dtype=np.uint8)

        # Paste patches
        for x, y, w, h, pred in patches:
            full_mask[y : y + h, x : x + w] = pred.astype(np.uint8)

        # Run-Length Encode
        rle = rle_encode(full_mask)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # Save to CSV
    if submission_rows:
        pd.DataFrame(submission_rows).to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print("No test data found. Submission file not generated.")


def train_model(limit_samples=None):
    """
    Main function to train the FR-UNet model.

    Args:
        limit_samples (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on {device}...")

    # 2. Data Loading
    train_dataset = InkDataset(mode="train", load_cached_data=True, limit=limit_samples)
    val_dataset = InkDataset(mode="val", load_cached_data=True, limit=limit_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model & Optimizer
    model = FRUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    best_f05 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, Config.POS_WEIGHT
        )
        val_loss, val_f05, val_thresh = validate(
            model, val_loader, device, Config.POS_WEIGHT
        )

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F0.5: {val_f05:.6f} (Thresh: {val_thresh:.2f})"
        )

        # Checkpoint & Early Stopping
        if val_f05 > best_f05:
            best_f05 = val_f05
            best_epoch = epoch
            patience_counter = 0

            # Save Model
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)

            # Save Threshold
            with open(
                os.path.join(Config.CHECKPOINT_DIR, "best_threshold.txt"), "w"
            ) as f:
                f.write(str(val_thresh))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training complete. Best F0.5: {best_f05:.6f} at Epoch {best_epoch+1}")

    # 5. Submission
    # Load best model weights
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Load best threshold
        thresh_path = os.path.join(Config.CHECKPOINT_DIR, "best_threshold.txt")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                best_thresh = float(f.read())
        else:
            best_thresh = 0.5

        generate_submission(model, device, best_thresh)
    else:
        print("No best model found. Skipping submission.")
