import os
import random
import numpy as np
import cv2
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates the object.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style) as per competition requirement
    pixels = mask.T.flatten()
    # Pad with 0s to detect changes at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoding (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

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


def pad_image_128(image):
    """
    Pads an image from (101, 101) to (128, 128) using reflection padding.

    Args:
        image (np.ndarray): Input image of shape (101, 101) or (101, 101, C).

    Returns:
        np.ndarray: Padded image of shape (128, 128) or (128, 128, C).
    """
    h, w = image.shape[:2]
    target_h, target_w = 128, 128

    pad_h = target_h - h
    pad_w = target_w - w

    # 27 pixels diff: 13 top/left, 14 bottom/right
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)


def unpad_image_101(image):
    """
    Crops an image from (128, 128) back to (101, 101).

    Args:
        image (np.ndarray): Padded image of shape (128, 128) or (128, 128, C).

    Returns:
        np.ndarray: Cropped image of shape (101, 101) or (101, 101, C).
    """
    h, w = image.shape[:2]
    target_h, target_w = 101, 101

    pad_h = h - target_h
    pad_w = w - target_w

    top = pad_h // 2
    left = pad_w // 2

    return image[top : top + target_h, left : left + target_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) over IoU thresholds [0.5, 0.95] with step 0.05.

    Args:
        predict (np.ndarray): Predictions. Can be probabilities or binary.
                              Shape (N, H, W) or (H, W).
        truth (np.ndarray): Ground truth masks. Binary (0/1).
                            Shape (N, H, W) or (H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.
                           Defaults to 0.5.

    Returns:
        float: The mean average precision score.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Handle single image input
    if predict.ndim == 2:
        predict = predict[np.newaxis, ...]
        truth = truth[np.newaxis, ...]

    # Binarize predictions
    # If already binary (0/1 integer), this preserves it.
    # If probabilities, this thresholds it.
    if (
        predict.dtype == float
        or predict.dtype == np.float32
        or predict.dtype == np.float64
    ):
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = truth.astype(np.uint8)

    N = predict.shape[0]
    ious = []

    # Calculate IoU for each image in batch
    for i in range(N):
        p = predict[i].flatten()
        t = truth[i].flatten()

        t_sum = t.sum()
        p_sum = p.sum()

        if t_sum == 0 and p_sum == 0:
            ious.append(1.0)
        elif t_sum == 0 and p_sum > 0:
            ious.append(0.0)
        elif t_sum > 0 and p_sum == 0:
            ious.append(0.0)
        else:
            intersection = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            ious.append(intersection / union)

    ious = np.array(ious)

    # Calculate Precision at each threshold
    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)

    # scores shape: (N, n_thresholds)
    # For each image, score is 1 if IoU > t, else 0 (True Positive vs False Positive/Negative)
    scores = (ious[:, None] > thresholds[None, :]).astype(float)

    # Average precision per image (over thresholds)
    image_aps = scores.mean(axis=1)

    # Mean over the batch
    return image_aps.mean()
