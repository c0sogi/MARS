import numpy as np
import torch
import random
import os


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Fortran-order).

    Args:
        mask (np.array): Binary mask of shape (H, W) where 1 indicates the object.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    The metric sweeps over a range of IoU thresholds with a step size of 0.05.
    For each image, the score is the mean of the precision values at each threshold.
    Precision at a threshold t is 1 if IoU > t, else 0 (for single-class segmentation).

    Args:
        predict (np.array or torch.Tensor): Predicted masks. Shape (N, H, W) or (N, 1, H, W).
                                            Can be probabilities (0-1) or binary.
        truth (np.array or torch.Tensor): Ground truth masks. Shape (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to convert probability predictions to binary masks.

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Squeeze channel dimension if present (N, 1, H, W) -> (N, H, W)
    if predict.ndim == 4:
        predict = predict.squeeze(1)
    if truth.ndim == 4:
        truth = truth.squeeze(1)

    # Convert probabilities to binary mask
    if (
        predict.dtype == float
        or predict.dtype == np.float32
        or predict.dtype == np.float64
    ):
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = truth.astype(np.uint8)

    ious = []

    # Calculate IoU per image
    for i in range(len(predict)):
        p = predict[i]
        t = truth[i]

        intersection = np.sum(p * t)
        union = np.sum(p) + np.sum(t) - intersection

        if union == 0:
            # Both empty: perfect match (IoU = 1.0)
            # One empty, one not: handled by intersection=0, union>0 -> IoU=0.0
            # Since union=0 implies both sum(p)=0 and sum(t)=0
            ious.append(1.0)
        else:
            ious.append(intersection / union)

    ious = np.array(ious)

    # Thresholds: 0.5, 0.55, 0.6, ..., 0.95
    thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)

    # Calculate score
    # For a single image, score is the fraction of thresholds that the IoU exceeds.
    # We broadcast comparison: (N_images, 1) > (1, N_thresholds)
    matches = ious[:, None] > thresholds[None, :]

    # Mean over thresholds for each image gives the AP per image
    image_scores = np.mean(matches, axis=1)

    # Return mean over the batch
    return np.mean(image_scores)
