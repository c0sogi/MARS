import os
import random
import numpy as np
import torch
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


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 0 for background, 1 for object.

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-major as per competition requirement
    pixels = mask.flatten(order="F")

    # We prepend and append 0 to detect runs at the beginning and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the pixel value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are starts, runs[1::2] are ends (exclusive)
    # The length is end - start.
    # Note: The +1 above gives us 1-based indexing for the starts naturally
    # because the indices in `runs` correspond to the index in the padded array,
    # which effectively shifts the original 0-based index by 1.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape of the mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # 0-indexed
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def compute_iou_batch(preds, labels):
    """
    Computes IoU for a batch of predictions and labels.

    Args:
        preds (np.ndarray): Binary predictions (B, H, W).
        labels (np.ndarray): Binary ground truth (B, H, W).

    Returns:
        np.ndarray: IoU scores for each item in the batch (shape: (B,)).
    """
    # Flatten spatial dimensions
    preds_flat = preds.reshape(preds.shape[0], -1) > 0
    labels_flat = labels.reshape(labels.shape[0], -1) > 0

    intersection = (preds_flat & labels_flat).sum(axis=1)
    union = (preds_flat | labels_flat).sum(axis=1)

    iou = np.ones(preds.shape[0], dtype=np.float32)

    # If union is > 0, calculate IoU. If union is 0 (both empty), IoU remains 1.0.
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    return iou


def compute_map_batch(preds, labels, thresholds=None):
    """
    Computes the Mean Average Precision (mAP) at specified IoU thresholds.

    Args:
        preds (np.ndarray): Binary predictions (B, H, W).
        labels (np.ndarray): Binary ground truth (B, H, W).
        thresholds (list or np.ndarray, optional): IoU thresholds.
                                                   Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        float: The mean average precision over the batch.
    """
    if thresholds is None:
        thresholds = Config.IOU_THRESHOLDS

    thresholds = np.array(thresholds)

    # Calculate IoU for each image in the batch
    ious = compute_iou_batch(preds, labels)

    # Compare IoUs to thresholds
    # ious: (B,) -> (B, 1)
    # thresholds: (T,) -> (1, T)
    # matches: (B, T) boolean matrix
    matches = ious[:, None] > thresholds[None, :]

    # Average precision per image is the mean of matches across thresholds
    # ap_per_image: (B,)
    ap_per_image = matches.mean(axis=1)

    # Mean AP over the batch
    return ap_per_image.mean()


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

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


def log_print(message):
    """
    Prints a message to console. Can be extended to log to file.
    """
    print(message)


def save_to_cache(filename, data):
    """
    Saves data to the cache directory using numpy format.

    Args:
        filename (str): Name of the file (e.g., 'processed_masks.npy').
        data (np.ndarray): Data to save.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.save(file_path, data)


def load_from_cache(filename):
    """
    Loads data from the cache directory.

    Args:
        filename (str): Name of the file to load.

    Returns:
        np.ndarray or None: The loaded data, or None if file does not exist.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        try:
            return np.load(file_path, allow_pickle=False)
        except ValueError:
            # Fallback if object arrays were saved (though discouraged)
            return np.load(file_path, allow_pickle=True)
    return None
