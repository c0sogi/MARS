import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric expects a space-delimited list of pairs (start, length).
    Pixels are numbered from left to right, then top to bottom (row-major), starting at 1.

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten the mask in row-major order (C-style)
    pixels = mask.flatten()

    # Prepend and append 0 to detect transitions at the start and end of the array
    # This simplifies finding the edges of the runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    # The result of np.where is a tuple, we take the first element [0]
    # We add 1 because the indices are 0-based, but we want 1-based pixel positions
    # (and the shift caused by prepending 0 aligns this naturally)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' now contains [start1, end1, start2, end2, ...]
    # The metric expects [start1, length1, start2, length2, ...]
    # Length = end - start
    # We update the odd indices (ends) to be lengths: end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def apply_tta(model, inputs):
    """
    Applies Test-Time Augmentation (TTA) to a batch of inputs.

    Augmentations applied based on Config:
    - Horizontal/Vertical Flips
    - 90-degree Rotations

    Args:
        model (nn.Module): The neural network model.
        inputs (torch.Tensor): Batch of inputs, shape (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability map, shape (B, 1, H, W).
    """
    preds = []

    # Helper to run inference and apply sigmoid
    def get_pred(x):
        with torch.no_grad():
            logits = model(x)
            return torch.sigmoid(logits)

    # 1. Original
    preds.append(get_pred(inputs))

    # 2. Flips
    if Config.TTA_FLIPS:
        # Horizontal Flip (dim 3 is width)
        x_h = torch.flip(inputs, dims=[3])
        p_h = get_pred(x_h)
        preds.append(torch.flip(p_h, dims=[3]))

        # Vertical Flip (dim 2 is height)
        x_v = torch.flip(inputs, dims=[2])
        p_v = get_pred(x_v)
        preds.append(torch.flip(p_v, dims=[2]))

    # 3. Rotations
    if Config.TTA_ROTATIONS:
        # k=1: 90 degrees
        x_r1 = torch.rot90(inputs, k=1, dims=[2, 3])
        p_r1 = get_pred(x_r1)
        preds.append(torch.rot90(p_r1, k=-1, dims=[2, 3]))

        # k=2: 180 degrees
        x_r2 = torch.rot90(inputs, k=2, dims=[2, 3])
        p_r2 = get_pred(x_r2)
        preds.append(torch.rot90(p_r2, k=-2, dims=[2, 3]))

        # k=3: 270 degrees
        x_r3 = torch.rot90(inputs, k=3, dims=[2, 3])
        p_r3 = get_pred(x_r3)
        preds.append(torch.rot90(p_r3, k=-3, dims=[2, 3]))

    # Stack all predictions and compute the mean
    preds = torch.stack(preds, dim=0)
    avg_pred = torch.mean(preds, dim=0)

    return avg_pred


def predict_tiled(model, volume, mask, mean, std, device, overlap_ratio=0.5):
    """
    Performs sliding-window inference on a large 3D volume.

    Args:
        model (nn.Module): Trained PyTorch model.
        volume (np.ndarray): Raw input volume, shape (Z, H, W).
        mask (np.ndarray): Binary mask indicating valid fragment area, shape (H, W).
        mean (float): Global mean for normalization.
        std (float): Global std for normalization.
        device (str): 'cuda' or 'cpu'.
        overlap_ratio (float): Fraction of overlap between tiles (0.0 to 1.0).

    Returns:
        np.ndarray: Probability map of the entire fragment, shape (H, W), float32.
    """
    model.eval()
    z_dim, h, w = volume.shape
    patch_size = Config.PATCH_SIZE

    # Calculate stride based on overlap
    stride = int(patch_size * (1 - overlap_ratio))
    if stride < 1:
        stride = 1

    # Initialize accumulators
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Generate grid coordinates
    # We ensure the last patch ends exactly at the border by adding a specific coordinate
    y_starts = list(range(0, h - patch_size + 1, stride))
    if (h - patch_size) % stride != 0:
        y_starts.append(h - patch_size)
    # Handle case where image is smaller than patch size (unlikely but safe)
    if not y_starts and h < patch_size:
        # Not handling upscaling for small images as per dataset specs (images are large)
        pass

    x_starts = list(range(0, w - patch_size + 1, stride))
    if (w - patch_size) % stride != 0:
        x_starts.append(w - patch_size)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                y_end = y + patch_size
                x_end = x + patch_size

                # Check validity using the mask
                # If the patch is completely outside the fragment, skip it
                patch_mask = mask[y:y_end, x:x_end]
                if not np.any(patch_mask):
                    continue

                # Extract volume patch
                # Shape: (65, patch_size, patch_size)
                vol_patch = volume[:, y:y_end, x:x_end]

                # Normalize on-the-fly
                # Convert to float32 only for the patch to save memory
                vol_patch = (vol_patch.astype(np.float32) - mean) / std

                # Prepare tensor
                # Add batch dimension: (1, 65, H, W)
                tensor = torch.from_numpy(vol_patch).unsqueeze(0).to(device)

                # Inference
                if Config.TTA_FLIPS or Config.TTA_ROTATIONS:
                    pred = apply_tta(model, tensor)
                else:
                    logits = model(tensor)
                    pred = torch.sigmoid(logits)

                # Accumulate result
                # pred is (1, 1, H, W) -> squeeze to (H, W)
                pred_np = pred.squeeze().cpu().numpy()

                prob_map[y:y_end, x:x_end] += pred_np
                count_map[y:y_end, x:x_end] += 1.0

    # Average the predictions in overlapping regions
    # Avoid division by zero
    valid_pixels = count_map > 0
    prob_map[valid_pixels] /= count_map[valid_pixels]

    # Zero out predictions outside the valid fragment mask
    prob_map = prob_map * (mask > 0)

    return prob_map
