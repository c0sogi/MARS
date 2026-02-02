import numpy as np
import torch
import os


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-major (Fortran-style) as per competition requirement
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

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
    Calculates the mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        truth (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision score.
    """
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    pred_mask = (predict > threshold).astype(np.uint8)
    truth_mask = (truth > 0.5).astype(np.uint8)

    # Handle single image case by adding batch dimension
    if pred_mask.ndim == 2:
        pred_mask = pred_mask[None, :, :]
        truth_mask = truth_mask[None, :, :]

    # Flatten spatial dimensions: (N, H, W) -> (N, H*W)
    p_flat = pred_mask.reshape(pred_mask.shape[0], -1)
    t_flat = truth_mask.reshape(truth_mask.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (p_flat & t_flat).sum(axis=1)
    union = (p_flat | t_flat).sum(axis=1)

    # Calculate IoU
    # If union is 0, it means both pred and truth are empty -> IoU = 1
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty_union = union > 0
    iou[non_empty_union] = intersection[non_empty_union] / union[non_empty_union]

    # Calculate Precision at thresholds
    # Thresholds: 0.5, 0.55, 0.6, ..., 0.95
    thresholds = np.arange(0.5, 0.96, 0.05)

    # Compare IoU against thresholds: (N, 1) > (1, 10) -> (N, 10)
    matches = iou[:, None] > thresholds[None, :]

    # Average over thresholds for each image
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return np.mean(image_scores)


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        score (float): Validation score.
        path (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": float(score),
        },
        path,
    )
