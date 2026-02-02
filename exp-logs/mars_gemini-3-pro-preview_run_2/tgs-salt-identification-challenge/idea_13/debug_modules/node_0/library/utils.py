import numpy as np
import cv2
import torch
from library.config import Config


def rle_encode(mask):
    """
    Encodes a mask in Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 0 for background, 1 for object.

    Returns:
        str: Space-delimited list of pairs (start_index, run_length).
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-major
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (every second element minus the previous)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE)):
    """
    Decodes a Run-Length Encoded string into a mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def pad_image(image):
    """
    Pads an image from ORIG_IMG_SIZE to IMG_SIZE using reflection.
    Handles both (H, W) and (H, W, C) inputs.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Padded image.
    """
    target_h = Config.IMG_SIZE
    target_w = Config.IMG_SIZE
    h, w = image.shape[:2]

    if h == target_h and w == target_w:
        return image

    pad_h = target_h - h
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top

    pad_w = target_w - w
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # cv2.copyMakeBorder handles multi-channel images automatically
    return cv2.copyMakeBorder(
        image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT
    )


def unpad_image(image, original_size=(Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE)):
    """
    Crops an image from IMG_SIZE back to original_size (center crop).

    Args:
        image (np.ndarray): Padded image.
        original_size (tuple): Target (H, W).

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    target_h, target_w = original_size

    if h == target_h and w == target_w:
        return image

    pad_h = h - target_h
    pad_top = pad_h // 2

    pad_w = w - target_w
    pad_left = pad_w // 2

    return image[pad_top : pad_top + target_h, pad_left : pad_left + target_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (np.ndarray or torch.Tensor): Predicted masks or probabilities.
                                              Shape (N, H, W) or (N, 1, H, W).
        truth (np.ndarray or torch.Tensor): Ground truth masks.
                                            Shape (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Squeeze channel dim if present
    if predict.ndim == 4:
        predict = predict.squeeze(1)
    if truth.ndim == 4:
        truth = truth.squeeze(1)

    # Binarize predictions if float
    if (
        predict.dtype == float
        or predict.dtype == np.float32
        or predict.dtype == np.float64
    ):
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = (truth > 0.5).astype(np.uint8)

    # Flatten spatial dimensions for batch IoU calculation
    # Shape becomes (N, H*W)
    p_flat = predict.reshape(predict.shape[0], -1)
    t_flat = truth.reshape(truth.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (p_flat & t_flat).sum(axis=1)
    union = (p_flat | t_flat).sum(axis=1)

    # Calculate IoU
    # If union is 0 (both empty), IoU is defined as 1
    iou = np.ones(predict.shape[0], dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoU to thresholds
    # iou shape: (N,)
    # thresholds shape: (10,)
    # matches shape: (N, 10)
    matches = iou[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    score_per_image = matches.mean(axis=1)

    # Mean over batch
    return score_per_image.mean()
