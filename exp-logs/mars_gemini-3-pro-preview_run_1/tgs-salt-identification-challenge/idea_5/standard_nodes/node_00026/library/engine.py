import torch
import numpy as np
from library.utils import compute_map_score


def center_crop(tensor, target_h=101, target_w=101):
    """
    Center crops a tensor to the target spatial dimensions.
    Assumes tensor format (N, C, H, W) or (N, H, W).
    """
    if tensor.dim() == 4:
        _, _, h, w = tensor.shape
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return tensor[:, :, start_h : start_h + target_h, start_w : start_w + target_w]
    elif tensor.dim() == 3:
        _, h, w = tensor.shape
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return tensor[:, start_h : start_h + target_h, start_w : start_w + target_w]
    return tensor


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Deep Supervision.
    """
    model.train()
    running_loss = 0.0

    for i, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        # In train mode, DeepResUNet returns a list: [out128, out64, out32]
        outputs = model(images)

        # Compute loss
        # DeepSupervisionLoss handles the list of outputs and target resizing
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Training Loss: {avg_loss}")

    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set using TTA (Horizontal Flip)
    and Center Cropping to original dimensions.
    Returns average loss and mAP score.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # --- Test-Time Augmentation (Horizontal Flip) ---

            # 1. Original Forward
            # In eval mode, DeepResUNet returns single tensor out128 (logits)
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Flipped Forward
            images_flipped = torch.flip(images, dims=[3])  # Flip width (dim 3 for NCHW)
            logits_flipped = model(images_flipped)

            # Unflip logits and probs
            logits_flipped_back = torch.flip(logits_flipped, dims=[3])
            probs_flipped_back = torch.sigmoid(logits_flipped_back)

            # 3. Average Predictions
            # Average logits for loss calculation (stable)
            avg_logits = (logits_orig + logits_flipped_back) / 2.0

            # Average probabilities for metric calculation
            avg_probs = (probs_orig + probs_flipped_back) / 2.0

            # --- Loss Calculation ---
            loss = criterion(avg_logits, masks)
            running_loss += loss.item()

            # --- Post-Processing (Center Crop) ---
            # Crop 128x128 -> 101x101 to match competition metric logic
            preds_cropped = center_crop(avg_probs, target_h=101, target_w=101)
            masks_cropped = center_crop(masks, target_h=101, target_w=101)

            # Accumulate for mAP calculation
            all_preds.append(preds_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())

    # Concatenate all batches
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)

    # Compute mAP
    # compute_map_score expects probabilities and handles thresholding internally
    map_score = compute_map_score(all_preds_np, all_targets_np)

    avg_loss = running_loss / len(dataloader)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation mAP: {map_score}")

    return avg_loss, map_score
