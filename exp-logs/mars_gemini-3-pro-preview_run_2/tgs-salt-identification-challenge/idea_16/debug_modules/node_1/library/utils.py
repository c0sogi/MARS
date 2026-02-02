import numpy as np


def rle_encode(img):
    """
    Convert a binary mask to Run-Length Encoding (RLE).

    Args:
        img (np.array): Binary mask image (0 for background, 1 for salt).
                        Shape (H, W).

    Returns:
        str: Space-delimited RLE string "start length start length ...".
    """
    # Flatten column-wise (Fortran-style)
    pixels = img.flatten(order="F")
    # Pad with 0s at ends to detect start/end of runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decode a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.array: Binary mask with shape `shape`.
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


def calculate_iou_batch(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for a batch of images.

    Args:
        y_true (np.array): Ground truth masks, shape (N, H, W) or (H, W).
        y_pred (np.array): Predicted masks, shape (N, H, W) or (H, W).

    Returns:
        np.array: IoU scores for each image, shape (N,).
    """
    # Ensure inputs are binary
    y_true = (y_true > 0.5).astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)

    # Handle single image case by adding batch dimension
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
        y_pred = y_pred[np.newaxis, ...]

    # Flatten spatial dimensions: (N, H*W)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    intersection = np.logical_and(y_true_flat, y_pred_flat).sum(axis=1)
    union = np.logical_or(y_true_flat, y_pred_flat).sum(axis=1)

    # Initialize IoU array
    iou = np.ones(y_true.shape[0], dtype=np.float32)

    # If union > 0, calculate IoU. If union == 0 (both empty), IoU remains 1.0.
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    return iou


def calculate_map(y_true, y_pred, thresholds=None):
    """
    Calculate Mean Average Precision (mAP) at multiple IoU thresholds.

    Args:
        y_true (np.array): Ground truth masks.
        y_pred (np.array): Predicted masks (binary or probabilities).
        thresholds (list/np.array, optional): IoU thresholds.
                                              Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        float: The mean average precision score.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)
    else:
        thresholds = np.array(thresholds)

    # Calculate IoU for each image in the batch
    ious = calculate_iou_batch(y_true, y_pred)

    # Compare IoUs to thresholds
    # ious: (N,) -> (N, 1)
    # thresholds: (T,) -> (1, T)
    # matches: (N, T) boolean matrix
    matches = ious[:, None] > thresholds[None, :]

    # Calculate precision per image (mean over thresholds)
    # Since it's binary segmentation, precision at threshold t is 1 if match else 0.
    avg_precision_per_image = matches.mean(axis=1)

    # Calculate mAP (mean over dataset)
    map_score = avg_precision_per_image.mean()

    return map_score
