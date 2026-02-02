import numpy as np
import torch


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 - salt, 0 - background.

    Returns:
        str: Space-delimited string of start positions and lengths.
    """
    # Flatten column-wise (Fortran-style) as required by the competition
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect starts and ends of runs efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    # The length of a run is end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calculate_iou_batch(y_pred, y_true, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) for a batch of images.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities or binary masks.
                                             Shape (N, H, W) or (N, 1, H, W).
        y_true (np.ndarray or torch.Tensor): Ground truth masks.
                                             Shape (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        np.ndarray: Array of IoU scores for each image in the batch. Shape (N,).
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Ensure shapes match and are flattened per image for vectorized calculation
    # Reshape to (N, -1) to handle spatial dimensions (H, W)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)

    # Binarize predictions and ensure labels are binary
    pred_mask = (y_pred > threshold).astype(np.uint8)
    true_mask = (y_true > 0.5).astype(np.uint8)

    # Calculate Intersection and Union
    intersection = np.sum(pred_mask * true_mask, axis=1)
    union = np.sum(pred_mask, axis=1) + np.sum(true_mask, axis=1) - intersection

    # Handle IoU calculation
    # Initialize IoU array. If union is 0, it means both pred and true are empty.
    # In this competition/metric, correctly predicting "nothing" is a perfect score (1.0).
    iou = np.ones_like(intersection, dtype=np.float64)

    # Only calculate division where union > 0
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    return iou


def _calculate_metric_from_iou(ious):
    """
    Internal helper to calculate the competition metric (mAP over thresholds)
    given a list of IoUs.

    The metric sweeps over IoU thresholds from 0.5 to 0.95 with step 0.05.
    At each threshold, a prediction is a "hit" if IoU > threshold.
    The score for an image is the mean of these binary hit/miss values.
    """
    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Vectorized comparison:
    # ious shape: (N,)
    # thresholds shape: (10,)
    # Result matches shape: (N, 10)
    matches = ious[:, None] > thresholds[None, :]

    # Mean over thresholds for each image -> Shape (N,)
    image_scores = np.mean(matches, axis=1)

    # Mean over the entire batch
    return np.mean(image_scores)


def optimize_threshold(y_pred, y_true, num_steps=50):
    """
    Finds the optimal binarization threshold that maximizes the mean Average Precision (mAP)
    on the provided validation set.

    Args:
        y_pred (np.ndarray or torch.Tensor): Validation predictions (probabilities).
        y_true (np.ndarray or torch.Tensor): Validation ground truths.
        num_steps (int): Number of threshold steps to search between 0.1 and 0.9.

    Returns:
        float: The optimal threshold value.
        float: The best mAP score achieved.
    """
    # Convert inputs to numpy once to avoid overhead in the loop
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    best_threshold = 0.5
    best_score = -1.0

    # Search range for probabilities.
    # We avoid 0.0 and 1.0 to prevent trivial all-0 or all-1 masks unless necessary.
    thresholds = np.linspace(0.1, 0.9, num_steps)

    for th in thresholds:
        # Calculate raw IoUs for this threshold
        ious = calculate_iou_batch(y_pred, y_true, threshold=th)

        # Calculate the specific competition metric
        score = _calculate_metric_from_iou(ious)

        if score > best_score:
            best_score = score
            best_threshold = th

    return best_threshold, best_score
