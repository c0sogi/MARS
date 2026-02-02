import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def sigmoid(x):
    """
    Applies the sigmoid function to the input.
    """
    return 1 / (1 + np.exp(-x))


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask: Numpy array of shape (H, W), where 1 indicates ink and 0 indicates background.

    Returns:
        String containing space-delimited start position and length pairs.
        Pixels are numbered from left to right, then top to bottom, 1-based.
    """
    pixels = mask.flatten()
    # Pad with zeros at start and end to define edges for runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-Beta score.

    Args:
        preds: Predictions (logits or probabilities). Tensor or Numpy array.
        targets: Ground truth labels. Tensor or Numpy array.
        beta: The beta parameter for the F-score (default 0.5 weights precision higher).
        threshold: Threshold for binarizing predictions.
        epsilon: Small constant to prevent division by zero.

    Returns:
        The computed F-Beta score.
    """
    if torch.is_tensor(preds):
        # Apply sigmoid if predictions are logits (outside [0, 1] range)
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        preds_bin = (preds > threshold).float()
        targets = targets.float()

        tp = (preds_bin * targets).sum()
        fp = (preds_bin * (1 - targets)).sum()
        fn = ((1 - preds_bin) * targets).sum()

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + epsilon)
        )
        return score
    else:
        # Numpy implementation
        if np.min(preds) < 0 or np.max(preds) > 1:
            preds = sigmoid(preds)

        preds_bin = (preds > threshold).astype(float)

        tp = (preds_bin * targets).sum()
        fp = (preds_bin * (1 - targets)).sum()
        fn = ((1 - preds_bin) * targets).sum()

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + epsilon)
        )
        return score


def dice_coef(preds, targets, smooth=1e-7):
    """
    Computes the Dice Coefficient.

    Args:
        preds: Predictions (logits or probabilities).
        targets: Ground truth labels.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        The computed Dice coefficient.
    """
    if torch.is_tensor(preds):
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return dice
    else:
        if np.min(preds) < 0 or np.max(preds) > 1:
            preds = sigmoid(preds)

        preds = preds.flatten()
        targets = targets.flatten()

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return dice


def normalize_image(image):
    """
    Normalizes an image using ImageNet mean and standard deviation.

    Args:
        image: Numpy array of shape (H, W, 3) with values in [0, 255].

    Returns:
        Normalized image of shape (H, W, 3).
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(Config.NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(Config.NORMALIZE_STD, dtype=np.float32)
    return (image - mean) / std


def denormalize_image(image):
    """
    Denormalizes an image for visualization purposes.

    Args:
        image: Normalized numpy array of shape (H, W, 3).

    Returns:
        Image array of shape (H, W, 3) with values in [0, 255], type uint8.
    """
    mean = np.array(Config.NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(Config.NORMALIZE_STD, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image * 255, 0, 255)
    return image.astype(np.uint8)


def save_visualization(image, mask, pred, save_path):
    """
    Saves a visualization of the input image, ground truth mask, and prediction side-by-side.

    Args:
        image: Input image array (H, W, 3) or (3, H, W).
        mask: Ground truth mask (H, W).
        pred: Predicted mask (H, W).
        save_path: File path to save the visualization.
    """
    # Handle channel-first format (PyTorch default)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Denormalize if the image is float (normalized)
    if image.dtype != np.uint8:
        image = denormalize_image(image)

    # Convert masks to 3-channel images for concatenation
    mask_vis = np.stack([mask] * 3, axis=-1) * 255
    pred_vis = np.stack([pred] * 3, axis=-1) * 255

    # Ensure correct data types
    mask_vis = mask_vis.astype(np.uint8)
    pred_vis = pred_vis.astype(np.uint8)

    # Concatenate horizontally: Image | Ground Truth | Prediction
    combined = np.hstack([image, mask_vis, pred_vis])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)
