import os
import numpy as np
import pydicom
import cv2
import torch


def read_dicom_windowed(path, window_center, window_width):
    """
    Reads a DICOM file, converts to Hounsfield Units, applies windowing,
    and normalizes to [0, 1].

    Args:
        path (str): Path to the DICOM file.
        window_center (int): Window center (level).
        window_width (int): Window width.

    Returns:
        np.ndarray: 2D numpy array of the image, normalized [0, 1].
    """
    if not os.path.exists(path):
        # Return a blank image if file is missing to prevent crash
        # Assuming 512x512 default DICOM size, but resizing happens later anyway
        return np.zeros((512, 512), dtype=np.float32)

    try:
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * slope + intercept

        # Apply Windowing
        min_val = window_center - window_width / 2.0
        max_val = window_center + window_width / 2.0

        # Clip values to window
        img = np.clip(pixel_array, min_val, max_val)

        # Normalize to [0, 1]
        if window_width > 0:
            img = (img - min_val) / window_width

        return img.astype(np.float32)

    except Exception as e:
        return np.zeros((512, 512), dtype=np.float32)


def load_2_5d_stack(
    image_dir, slice_filenames, slice_index, window_center, window_width, img_size
):
    """
    Loads a 2.5D stack of images (slice i-1, i, i+1) for a given study.
    Handles boundary conditions by replicating the first/last slice.

    Args:
        image_dir (str): Directory containing the study's DICOM files.
        slice_filenames (list): Sorted list of filenames (e.g. ['1.dcm', '2.dcm', ...]).
        slice_index (int): Index of the center slice to load.
        window_center (int): Window center for preprocessing.
        window_width (int): Window width for preprocessing.
        img_size (tuple): Target size (height, width).

    Returns:
        np.ndarray: Stacked image array of shape (H, W, 3).
    """
    num_slices = len(slice_filenames)

    # Identify indices for i-1, i, i+1 with boundary clamping
    # If index is 0, use [0, 0, 1]
    # If index is last, use [N-2, N-1, N-1]
    idx_prev = max(0, slice_index - 1)
    idx_curr = slice_index
    idx_next = min(num_slices - 1, slice_index + 1)

    indices = [idx_prev, idx_curr, idx_next]
    channels = []

    for idx in indices:
        filename = slice_filenames[idx]
        path = os.path.join(image_dir, filename)

        # Load and preprocess
        img = read_dicom_windowed(path, window_center, window_width)

        # Resize
        # cv2.resize expects (width, height), img_size is usually (height, width)
        # Assuming square or handling correctly
        if img.shape[:2] != img_size:
            img = cv2.resize(
                img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR
            )

        channels.append(img)

    # Stack along the last dimension -> (H, W, 3)
    stack = np.stack(channels, axis=-1)

    return stack


def weighted_log_loss(y_true, y_pred, weights=None):
    """
    Calculates the weighted multi-label logarithmic loss.

    Formula: L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    Loss is averaged across all rows (all predictions).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, 8).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, 8).
        weights (list or np.ndarray, optional): Weights for each of the 8 columns.
            Defaults to [1, 1, 1, 1, 1, 1, 1, 7] (C1-C7, patient_overall).

    Returns:
        float: The weighted log loss.
    """
    # Ensure inputs are numpy arrays
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Default weights: C1-C7 = 1, patient_overall = 7
    # This weights the 'any' label equal to the sum of all specific vertebrae
    if weights is None:
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    else:
        weights = np.array(weights)

    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate binary cross entropy
    # shape: (N, 8)
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights
    # weights broadcasts to (N, 8)
    weighted_bce = bce * weights

    # Average over all entries (rows and columns)
    # The competition metric averages "across all rows" in the submission file.
    # Since each column in our (N, 8) matrix corresponds to a row in the submission,
    # taking the global mean is correct.
    return np.mean(weighted_bce)


def sort_filenames_numerically(filenames):
    """
    Sorts a list of filenames based on the integer value of the filename
    (e.g., '10.dcm' comes after '1.dcm', not before '2.dcm').
    """
    return sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
