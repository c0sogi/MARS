import os
import random
import numpy as np
import torch
import cv2
import pydicom
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def recalibrate_prediction(
    probs, train_prior=Config.POSITIVE_SAMPLING_RATIO, target_prior=Config.TARGET_PRIOR
):
    """
    Recalibrates probabilities from a balanced training distribution to the target distribution.
    Cite {solution_lesson_node_00001}

    Args:
        probs (np.ndarray): Array of predicted probabilities.
        train_prior (float): Prior probability in the training set (sampling ratio).
        target_prior (float): Prior probability in the target/test set.

    Returns:
        np.ndarray: Recalibrated probabilities.
    """
    probs = np.array(probs)

    # Avoid numerical instability
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Convert to odds
    odds = probs / (1 - probs)

    # Calculate correction factor
    # Odds_corrected = Odds_predicted * (P_test / (1-P_test)) / (P_train / (1-P_train))
    factor = (target_prior * (1 - train_prior)) / (train_prior * (1 - target_prior))

    # Apply correction
    new_odds = odds * factor

    # Convert back to probability
    new_probs = new_odds / (1 + new_odds)

    return new_probs


def load_dicom_image(path, img_size=Config.IMG_SIZE):
    """
    Loads a DICOM image, handles PhotometricInterpretation, normalizes to [0, 1],
    and resizes to the target size.

    Args:
        path (str): Path to the .dcm file.
        img_size (tuple): Target size (Height, Width). Defaults to Config.IMG_SIZE.

    Returns:
        np.ndarray: Preprocessed image as a float32 numpy array with values in [0, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    try:
        # Read the DICOM file
        # pydicom handles pixel data extraction
        dcm = pydicom.dcmread(path)

        # Access pixel data
        img = dcm.pixel_array

        # Handle Photometric Interpretation
        # If MONOCHROME1, 0 is white (dense) and max is black.
        # We want 0 to be background (black) and max to be dense (white).
        # So we invert MONOCHROME1 images.
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Convert to float32
        img = img.astype(np.float32)

        # Min-Max Normalization to [0, 1]
        # Avoid division by zero if image is constant
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Resize
        if img_size is not None:
            # cv2.resize expects (width, height)
            # img_size is (Height, Width)
            height, width = img_size
            img = cv2.resize(img, (width, height))

        return img

    except Exception as e:
        # Raise error to ensure data issues are caught
        raise RuntimeError(f"Error loading DICOM file {path}: {e}")
