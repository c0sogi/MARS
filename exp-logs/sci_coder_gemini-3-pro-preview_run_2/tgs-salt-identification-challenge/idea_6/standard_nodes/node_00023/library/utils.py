import numpy as np


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.
    The format is a space-delimited list of pairs (start_index, run_length).
    Pixels are 1-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           Values should be 0 or 1.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style) to match the specification
    pixels = mask.flatten(order="F")

    # Pad with 0s at both ends to detect start and end of runs easily
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes an RLE string back to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if str(mask_rle) == "nan" or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (column-major)
    return img.reshape(shape, order="F")


def calc_iou(pred, target):
    """
    Calculates the Intersection over Union (IoU) between two masks.

    Args:
        pred (np.ndarray): Predicted mask (H, W). Can be probabilities or binary.
        target (np.ndarray): Ground truth mask (H, W).

    Returns:
        float: IoU score.
    """
    # Binarize predictions and targets
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)

    intersection = (pred & target).sum()
    union = (pred | target).sum()

    # Handle empty masks case
    if union == 0:
        # If both are empty, IoU is defined as 1.0 (perfect match of "nothing")
        return 1.0

    return intersection / union


def calc_map(preds, targets):
    """
    Calculates the Mean Average Precision (mAP) over a range of IoU thresholds.
    Thresholds: 0.5, 0.55, ..., 0.95.

    Args:
        preds (list or np.ndarray): List of predicted masks.
        targets (list or np.ndarray): List of ground truth masks.

    Returns:
        float: Mean Average Precision score.
    """
    thresholds = np.arange(0.5, 0.96, 0.05)
    scores = []

    for p, t in zip(preds, targets):
        iou = calc_iou(p, t)

        # At each threshold t, precision is 1 if IoU > t, else 0.
        # The score for a single image is the mean of these precisions.
        matches = iou > thresholds
        score = np.mean(matches)
        scores.append(score)

    return np.mean(scores)
