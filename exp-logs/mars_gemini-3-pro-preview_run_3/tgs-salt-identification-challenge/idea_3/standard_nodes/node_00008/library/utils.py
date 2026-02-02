import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle is None or str(mask_rle) == "nan" or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def iou_metric(y_pred_bin, y_true_bin, eps=1e-7):
    """
    Calculate raw IoU for a batch of binary masks.
    y_pred_bin: (Batch, H, W)
    y_true_bin: (Batch, H, W)
    Returns: float (mean IoU)
    """
    # Flatten spatial dimensions
    # Ensure inputs are integer type for bitwise operations
    y_pred_bin = y_pred_bin.astype(np.uint8)
    y_true_bin = y_true_bin.astype(np.uint8)

    pred_flat = y_pred_bin.reshape(y_pred_bin.shape[0], -1)
    true_flat = y_true_bin.reshape(y_true_bin.shape[0], -1)

    intersection = (pred_flat & true_flat).sum(axis=1)
    union = (pred_flat | true_flat).sum(axis=1)

    # Handle empty union (both empty) -> IoU = 1
    iou = np.ones(y_pred_bin.shape[0])
    mask = union > 0
    iou[mask] = intersection[mask] / (union[mask] + eps)

    return iou.mean()


def calc_map(preds, targets, threshold=0.5):
    """
    Calculate Mean Average Precision at IoU thresholds (0.5 to 0.95).
    preds: Tensor or Numpy array (Batch, H, W) or (Batch, 1, H, W). Probabilities.
    targets: Tensor or Numpy array (Batch, H, W) or (Batch, 1, H, W). 0 or 1.
    threshold: Threshold to binarize preds.
    """
    # Convert to numpy if tensor
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Squeeze channel dim if present
    if preds.ndim == 4 and preds.shape[1] == 1:
        preds = preds.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    # Binarize predictions and ensure targets are binary
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    # Flatten spatial dims: (Batch, -1)
    batch_size = preds_bin.shape[0]
    preds_bin = preds_bin.reshape(batch_size, -1)
    targets_bin = targets_bin.reshape(batch_size, -1)

    # Calculate Intersection and Union
    intersection = (preds_bin & targets_bin).sum(axis=1)
    union = (preds_bin | targets_bin).sum(axis=1)

    # Calculate IoU
    # Handle empty masks: if union is 0, it means both are empty -> IoU = 1
    iou = np.ones(batch_size, dtype=np.float32)
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    # Thresholds: 0.5, 0.55, ..., 0.95
    # np.arange(0.5, 1.0, 0.05) gives [0.5, 0.55, ..., 0.95]
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoU to thresholds
    # iou: (batch,)
    # thresholds: (10,)
    # Result: (batch, 10)
    # Metric definition: IoU > threshold
    matches = iou[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    # Since we treat the mask as a single object:
    # If match (IoU > t), precision is 1. If no match, precision is 0.
    avg_precision_per_image = matches.mean(axis=1)

    # Mean over batch
    return avg_precision_per_image.mean()
