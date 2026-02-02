import os
import random
import numpy as np
import torch


class CFG:
    """
    Configuration class for hyperparameters and paths.
    """

    # General
    seed = 42
    debug = False
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Paths
    input_root = "./input"
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"
    cache_dir = "./working/idea_2/"

    # Data Parameters
    img_size = 512

    # Model Parameters
    # Using EfficientNet-B4 as backbone for U-Net++
    backbone = "efficientnet_b4"
    encoder_weights = "imagenet"
    num_classes = 1

    # Training Parameters
    epochs = 15
    batch_size = 8  # Adjusted for 512x512 resolution on T4
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    scheduler = "CosineAnnealingWarmRestarts"
    T_0 = 5  # For CosineAnnealingWarmRestarts
    T_mult = 1

    # Inference
    threshold = 0.5


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask where 1 indicates the object and 0 background.

    Returns:
        str: Space-separated run-length encoding.
    """
    # Flatten column-wise (Fortran-style) as per competition requirement:
    # "pixels are numbered from top to bottom, then left to right"
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Computes the Dice Coefficient between ground truth and prediction.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient score.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )
