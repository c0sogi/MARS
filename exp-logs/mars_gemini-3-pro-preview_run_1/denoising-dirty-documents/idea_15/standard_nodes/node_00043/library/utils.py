import os
import random
import numpy as np
import torch
import cv2
from library.config import GlobalConfig


def seed_everything(seed=GlobalConfig.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Ensures deterministic behavior for the ensemble training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_image(image, modulus=GlobalConfig.PAD_MODULUS):
    """
    Applies reflection padding to the image such that its dimensions are multiples of modulus.
    Pads to the bottom and right to maintain coordinate simplicity.

    Args:
        image (np.ndarray): Input image (H, W) or (H, W, C).
        modulus (int): The divisor the dimensions must be multiples of.

    Returns:
        tuple: (padded_image, padding_info)
               padding_info is (top, bottom, left, right)
    """
    h, w = image.shape[:2]
    pad_h = (modulus - h % modulus) % modulus
    pad_w = (modulus - w % modulus) % modulus

    # Pad bottom and right using reflection
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    # Return padding info to allow cropping/mapping later if needed
    return padded_image, (0, pad_h, 0, pad_w)


def d4_transform(image, k):
    """
    Applies the k-th geometric transformation from the D4 dihedral group.
    Used for Test-Time Augmentation (TTA).

    Args:
        image (np.ndarray): Input image.
        k (int): Transformation index (0-7).

    Returns:
        np.ndarray: Transformed image (contiguous).
    """
    if k == 0:
        out = image
    elif k == 1:
        out = np.rot90(image, 1)
    elif k == 2:
        out = np.rot90(image, 2)
    elif k == 3:
        out = np.rot90(image, 3)
    elif k == 4:
        out = np.fliplr(image)
    elif k == 5:
        # Rot90(FlipH)
        out = np.rot90(np.fliplr(image), 1)
    elif k == 6:
        # Rot180(FlipH)
        out = np.rot90(np.fliplr(image), 2)
    elif k == 7:
        # Rot270(FlipH)
        out = np.rot90(np.fliplr(image), 3)
    else:
        raise ValueError("k must be between 0 and 7")

    return np.ascontiguousarray(out)


def d4_inverse_transform(image, k):
    """
    Applies the inverse of the k-th geometric transformation.
    Used to aggregate TTA predictions back to the original orientation.

    Args:
        image (np.ndarray): Transformed image (prediction).
        k (int): Transformation index (0-7) that was applied to the input.

    Returns:
        np.ndarray: Restored image (contiguous).
    """
    if k == 0:
        out = image
    elif k == 1:
        # Inverse of Rot90 is Rot270
        out = np.rot90(image, 3)
    elif k == 2:
        # Inverse of Rot180 is Rot180
        out = np.rot90(image, 2)
    elif k == 3:
        # Inverse of Rot270 is Rot90
        out = np.rot90(image, 1)
    elif k == 4:
        # Inverse of FlipH is FlipH
        out = np.fliplr(image)
    elif k == 5:
        # Inverse of Rot90(FlipH) -> FlipH(Rot270)
        out = np.fliplr(np.rot90(image, 3))
    elif k == 6:
        # Inverse of Rot180(FlipH) -> FlipH(Rot180)
        out = np.fliplr(np.rot90(image, 2))
    elif k == 7:
        # Inverse of Rot270(FlipH) -> FlipH(Rot90)
        out = np.fliplr(np.rot90(image, 1))
    else:
        raise ValueError("k must be between 0 and 7")

    return np.ascontiguousarray(out)
