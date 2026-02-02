import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The competition specifies that pixels are numbered from left to right,
    then top to bottom (Row-Major order).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten in Row-Major order (default for numpy)
    pixels = mask.flatten()

    # Prepend and append 0 to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_loss(pred, target, smooth=1e-6):
    """
    Calculates the Dice Loss for binary segmentation.

    Args:
        pred (torch.Tensor): Logits from the model of shape (B, 1, H, W).
        target (torch.Tensor): Binary ground truth of shape (B, 1, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Scalar Dice loss (1 - Dice Coefficient).
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def fbeta_score(pred, target, beta=0.5, threshold=0.5, smooth=1e-6):
    """
    Calculates the F-beta score (F0.5 for this competition).

    Args:
        pred (torch.Tensor): Logits from the model of shape (B, 1, H, W).
        target (torch.Tensor): Binary ground truth of shape (B, 1, H, W).
        beta (float): Beta value for F-score (0.5 weights precision higher).
        threshold (float): Threshold to binarize predictions.
        smooth (float): Smoothing factor.

    Returns:
        float: The F-beta score.
    """
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()

    # Flatten
    pred_bin = pred_bin.view(-1)
    target = target.view(-1)

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + smooth)

    return fbeta.item()


def apply_tta(model, x):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions across
    8 combinations of flips and rotations (D4 symmetry group).

    Args:
        model (torch.nn.Module): The segmentation model.
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability map of shape (B, 1, H, W).
    """
    probs = []

    # Define augmentations: (flip_horizontal, num_rotations_90)
    augmentations = [
        (False, 0),
        (False, 1),
        (False, 2),
        (False, 3),
        (True, 0),
        (True, 1),
        (True, 2),
        (True, 3),
    ]

    for flip, k in augmentations:
        aug_x = x

        # Apply geometric transformations
        if flip:
            aug_x = torch.flip(aug_x, dims=[3])
        if k > 0:
            aug_x = torch.rot90(aug_x, k=k, dims=[2, 3])

        # Inference
        with torch.no_grad():
            logits = model(aug_x)
            pred_prob = torch.sigmoid(logits)

        # Reverse transformations
        if k > 0:
            pred_prob = torch.rot90(pred_prob, k=-k, dims=[2, 3])
        if flip:
            pred_prob = torch.flip(pred_prob, dims=[3])

        probs.append(pred_prob)

    # Stack and average probabilities
    probs = torch.stack(probs)
    avg_prob = torch.mean(probs, dim=0)

    return avg_prob


def predict_tiled(
    model,
    volume,
    patch_size=Config.PATCH_SIZE,
    stride=Config.STRIDE,
    device=Config.DEVICE,
    mean=None,
    std=None,
):
    """
    Performs deterministic tiled inference on a large 3D volume using a sliding window approach.
    Handles normalization, padding, and stitching of overlapping patches.

    Args:
        model (torch.nn.Module): Trained model.
        volume (np.ndarray): 3D input volume of shape (Z, H, W).
        patch_size (int): Spatial size of patches (H, W).
        stride (int): Stride for sliding window.
        device (torch.device): Device to run inference on.
        mean (float, optional): Global mean for normalization. If None, calculated from volume.
        std (float, optional): Global std for normalization. If None, calculated from volume.

    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    model.eval()
    z_dim, h, w = volume.shape

    # Handle Normalization Stats
    if mean is None:
        mean = np.mean(volume)
    if std is None:
        std = np.std(volume)
    if std == 0:
        std = 1.0

    # Initialize accumulators
    prediction_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # Sliding window
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            # Extract patch
            patch = volume[:, y:y_end, x:x_end].astype(np.float32)

            # Normalize
            patch = (patch - mean) / std

            # Pad to square patch_size (required for rotation TTA)
            pad_h = patch_size - patch.shape[1]
            pad_w = patch_size - patch.shape[2]

            if pad_h > 0 or pad_w > 0:
                patch = np.pad(
                    patch,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Convert to tensor
            patch_tensor = (
                torch.from_numpy(patch).unsqueeze(0).to(device)
            )  # (1, Z, H, W)

            # Inference with TTA
            pred_prob = apply_tta(model, patch_tensor)

            # Convert back to numpy
            pred_patch = pred_prob.squeeze().cpu().numpy()  # (H, W)

            # Crop padding to restore original patch dimensions
            valid_h = y_end - y
            valid_w = x_end - x
            pred_patch = pred_patch[:valid_h, :valid_w]

            # Accumulate predictions
            prediction_map[y:y_end, x:x_end] += pred_patch
            count_map[y:y_end, x:x_end] += 1.0

    # Average overlapping predictions
    # count_map should not be zero anywhere given the stride logic, but safe division is good practice
    np.divide(prediction_map, count_map, out=prediction_map, where=count_map > 0)

    return prediction_map
