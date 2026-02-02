import os
import shutil
import numpy as np
import torch
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask of shape (H, W).
                          1 - salt, 0 - background.

    Returns:
        str: Space-delimited string of pairs (start, length).
    """
    # Flatten column-wise (F-order) as per competition requirement
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the competition metric: Mean Average Precision at different IoU thresholds.

    The metric sweeps over IoU thresholds from 0.5 to 0.95 with a step of 0.05.

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        truth (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize the predicted probabilities.

    Returns:
        float: The mean average precision score.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    # Flatten for IoU calculation per image
    # We assume inputs are (Batch, H, W) or (Batch, C, H, W)
    # If (H, W) add batch dim
    if predict.ndim == 2:
        predict = predict.reshape(1, -1)
        truth = truth.reshape(1, -1)
    else:
        predict = predict.reshape(predict.shape[0], -1)
        truth = truth.reshape(truth.shape[0], -1)

    intersection = (predict & truth).sum(axis=1)
    union = predict.sum(axis=1) + truth.sum(axis=1) - intersection

    # Calculate IoU
    # Handle division by zero: if union is 0, it means both pred and truth are empty.
    # In that case, IoU is 1.0.
    iou = np.zeros(predict.shape[0], dtype=np.float32)

    # Mask for non-empty union
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Mask for empty union (both empty) -> IoU = 1
    empty = union == 0
    iou[empty] = 1.0

    # Calculate score over thresholds
    # Thresholds: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    scores = []
    for t in thresholds:
        # Precision at threshold t is 1 if IoU > t, else 0
        # (Since there is effectively 1 object per image)
        tp = (iou > t).astype(np.float32)
        scores.append(tp)

    # Average over thresholds
    # shape: (num_thresholds, batch_size)
    scores = np.stack(scores, axis=0)

    # Mean per image
    image_scores = np.mean(scores, axis=0)

    # Mean over batch
    return np.mean(image_scores)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)
