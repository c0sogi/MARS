import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.utils import rle_encode, calculate_map
from library.losses import CombinedLoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def center_crop(tensor, target_h=101, target_w=101):
    """
    Center crops a tensor from (..., H, W) to (..., target_h, target_w).
    Assumes H, W >= target.
    """
    if tensor.ndim < 2:
        return tensor

    h, w = tensor.shape[-2:]
    diff_h = h - target_h
    diff_w = w - target_w

    # Calculate start indices (top-left)
    # Using integer division. If odd difference, this aligns with standard center crop logic.
    start_h = diff_h // 2
    start_w = diff_w // 2

    return tensor[..., start_h : start_h + target_h, start_w : start_w + target_w]


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Unpack batch: image, mask, depth, id
        images = batch[0].to(device)
        masks = batch[1].to(device)
        depths = batch[2].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (x, z)
        logits = model(images, depths)

        # Loss calculation
        loss = loss_fn(logits, masks)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Calculates Loss and performs linear search for best mAP threshold.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            masks = batch[1].to(device)
            depths = batch[2].to(device)

            batch_size = images.size(0)

            logits = model(images, depths)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Center crop back to 101x101 for metric calculation
            # Masks in loader are padded to 128x128, so we crop them too to compare valid regions
            probs_cropped = center_crop(probs, 101, 101)
            masks_cropped = center_crop(masks, 101, 101)

            # Move to CPU/Numpy for metric calculation
            all_preds.append(probs_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    # Shape: (N, 1, 101, 101) -> (N, 101, 101)
    y_pred = np.concatenate(all_preds, axis=0).squeeze(1)
    y_true = np.concatenate(all_targets, axis=0)

    # Threshold Optimization
    # Search range: 0.3 to 0.7 step 0.05
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_map = -1.0
    best_thresh = 0.5

    for t in thresholds:
        # Binarize
        pred_binary = (y_pred > t).astype(np.uint8)
        # Calculate mAP
        score = calculate_map(y_true, pred_binary)
        if score > best_map:
            best_map = score
            best_thresh = t

    return val_loss, best_map, best_thresh


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (Horizontal Flip).
    Enforces depth=0 (Generalist Mode) for all test images.
    """
    model.eval()
    predictions = []
    ids_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Test loader returns: image, depth, id
            images = batch[0].to(device)
            # Ignore loaded depth, enforce 0.0 for Generalist mode
            depths = torch.zeros(
                (images.size(0), 1), dtype=torch.float32, device=device
            )
            ids = batch[2]

            # 1. Forward Pass (Original)
            logits_orig = model(images, depths)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward Pass (Flipped)
            images_flipped = torch.flip(images, dims=[3])  # Flip width dimension
            logits_flip = model(images_flipped, depths)
            probs_flip = torch.sigmoid(logits_flip)
            # Flip back
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average probabilities
            probs_avg = (probs_orig + probs_flip_back) / 2.0

            # Center crop to 101x101
            probs_cropped = center_crop(probs_avg, 101, 101)

            # Store
            predictions.append(probs_cropped.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate
    predictions = np.concatenate(predictions, axis=0).squeeze(1)  # (N, 101, 101)
    return predictions, ids_list


def generate_submission(predictions, ids, threshold, output_path):
    """
    Generates the submission CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rle_masks = []
    for i in range(len(predictions)):
        pred_mask = (predictions[i] > threshold).astype(np.uint8)
        rle = rle_encode(pred_mask)
        rle_masks.append(rle)

    df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model(
    model, train_loader, val_loader, optimizer, device, num_epochs, patience, output_dir
):
    """
    Orchestrates the training process with Early Stopping.
    """
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pth")
    loss_fn = CombinedLoss()

    best_map = 0.0
    patience_counter = 0
    best_threshold = 0.5

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_loss, val_map, val_thresh = evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mAP: {val_map:.6f} | "
            f"Best Thresh: {val_thresh:.2f}"
        )

        # Early Stopping Check
        if val_map > best_map:
            best_map = val_map
            best_threshold = val_thresh
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best mAP: {best_map:.6f} at threshold {best_threshold:.2f}"
    )
    return best_threshold
