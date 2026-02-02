import os
import torch
import numpy as np
import pandas as pd
from library.utils import get_score, rle_encode, set_seed
from library.dataset import unpad_image


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch using strict FP32 precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for images, masks, depths, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (FP32)
        outputs = model(images, depths)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and mAP (at threshold 0.5 on logits).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for images, masks, depths, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Forward
            outputs = model(images, depths)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Prepare for scoring
            # Outputs are logits. We unpad them to 101x101.
            # Masks are 128x128. We unpad them to 101x101.

            # Convert to numpy
            logits_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(logits_np.shape[0]):
                # Unpad takes (H, W) or (H, W, C)
                # logits_np[i] is (1, 128, 128) -> (128, 128)
                pred_unpadded = unpad_image(logits_np[i, 0])
                mask_unpadded = unpad_image(masks_np[i, 0])

                all_preds.append(pred_unpadded)
                all_masks.append(mask_unpadded)

    epoch_loss = running_loss / dataset_size

    # Calculate mAP
    # We pass logits. get_score with threshold_value=None uses preds > 0.
    # This effectively thresholds logits at 0 (prob 0.5).
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    # Ensure masks are binary (should be already, but safe to cast)
    all_masks = (all_masks > 0).astype(np.uint8)

    map_score = get_score(all_preds, all_masks, threshold_value=None)

    return epoch_loss, map_score


def optimize_threshold(model, dataloader, device):
    """
    Finds the optimal probability threshold by sweeping over the validation set.
    Returns the best threshold.
    """
    model.eval()
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, _ in dataloader:
            images = images.to(device)
            depths = depths.to(device)

            # Forward
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()  # (B, 1, H, W)

            for i in range(probs_np.shape[0]):
                p = unpad_image(probs_np[i, 0])
                m = unpad_image(masks_np[i, 0])
                all_probs.append(p)
                all_masks.append(m)

    all_probs = np.array(all_probs)
    all_masks = (np.array(all_masks) > 0).astype(np.uint8)

    best_score = -1
    best_thresh = 0.5

    # Sweep
    thresholds = np.arange(0.3, 0.75, 0.05)
    for t in thresholds:
        # get_score handles thresholding if value is passed
        score = get_score(all_probs, all_masks, threshold_value=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(
        f"Threshold Optimization: Best mAP {best_score:.4f} at threshold {best_thresh:.2f}"
    )
    return best_thresh


def predict_marginalized(
    model, dataloader, device, depth_values=[-1.5, -0.75, 0.0, 0.75, 1.5]
):
    """
    Performs inference using Depth Scan Marginalization and TTA.
    Returns list of IDs and array of probability maps (N, 101, 101).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in dataloader:
            images = images.to(device)
            B = images.size(0)
            H, W = images.size(2), images.size(3)

            # Accumulator (B, 1, H, W)
            avg_probs = torch.zeros((B, 1, H, W), device=device)

            for z_val in depth_values:
                # Create depth batch
                z_tensor = torch.full((B, 1), z_val, device=device)

                # 1. Original
                logits = model(images, z_tensor)
                probs = torch.sigmoid(logits)

                # 2. Flip TTA
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped, z_tensor)
                probs_flipped = torch.sigmoid(logits_flipped)
                probs_flipped = torch.flip(probs_flipped, dims=[3])

                # Average TTA
                probs_z = (probs + probs_flipped) * 0.5

                # Accumulate
                avg_probs += probs_z

            # Average over depths
            avg_probs /= len(depth_values)

            # Process batch
            avg_probs_np = avg_probs.cpu().numpy()

            for i in range(B):
                # Unpad to 101x101
                pred_map = unpad_image(avg_probs_np[i, 0])
                all_preds.append(pred_map)
                all_ids.append(ids[i])

    return all_ids, np.array(all_preds)


def generate_submission(
    model, dataloader, device, output_path="./submission/submission.csv", threshold=0.5
):
    """
    Generates submission file using marginalized inference.
    """
    print(f"Generating submission with threshold {threshold}...")

    ids, probs = predict_marginalized(model, dataloader, device)

    # Binarize
    binary_masks = (probs > threshold).astype(np.uint8)

    # RLE Encode
    rle_masks = []
    for mask in binary_masks:
        rle_masks.append(rle_encode(mask))

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
