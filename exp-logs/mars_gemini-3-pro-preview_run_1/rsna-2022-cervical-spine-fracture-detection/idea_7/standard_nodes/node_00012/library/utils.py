import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom

# Try importing nibabel for NIFTI handling
try:
    import nibabel as nib
    import nibabel.orientations as nio

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from library.config import Config

# --------------------------------------------------------------------------
# DICOM & Image Processing
# --------------------------------------------------------------------------


def load_dicom_array(
    path, size=None, crop_roi=None, window_center=400, window_width=1800
):
    """
    Reads a DICOM file, applies windowing, and optionally resizes or crops.

    Args:
        path (str): Path to the .dcm file.
        size (tuple, optional): (height, width) to resize to.
        crop_roi (tuple, optional): (y, x, h, w) for cropping. Applied before resizing.
        window_center (int): Window center for CT.
        window_width (int): Window width for CT.

    Returns:
        np.ndarray: Processed 2D image array (float32, 0-1 range).
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Apply Rescale Slope/Intercept if present
        slope = getattr(dicom, "RescaleSlope", 1.0)
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Apply Windowing
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to 0-1
        img = (img - img_min) / (img_max - img_min)

        # Crop if requested
        if crop_roi is not None:
            y, x, h, w = crop_roi
            # Ensure bounds
            img_h, img_w = img.shape
            y = max(0, min(y, img_h - 1))
            x = max(0, min(x, img_w - 1))
            h = min(h, img_h - y)
            w = min(w, img_w - x)
            img = img[y : y + h, x : x + w]

        # Resize if requested
        if size is not None:
            if img.shape != size:
                img = cv2.resize(
                    img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR
                )

        return img

    except Exception as e:
        # Fallback for corrupt files or read errors
        # print(f"Error loading DICOM {path}: {e}")
        if size is not None:
            return np.zeros(size, dtype=np.float32)
        return np.zeros((512, 512), dtype=np.float32)


def load_scan_volume(study_uid, images_dir, size=None, load_cached_data=True):
    """
    Loads all DICOM slices for a study, forming a 3D volume.
    Implements caching to .npy files.

    Args:
        study_uid (str): Study Instance UID.
        images_dir (str): Directory containing study folders (e.g. train_images).
        size (tuple, optional): Resize dimensions for each slice (H, W).
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        np.ndarray: 3D volume (Depth, Height, Width).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_uid}.npy")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Failed to load cache, recompute

    # 2. Compute from Scratch
    study_dir = os.path.join(images_dir, study_uid)
    if not os.path.exists(study_dir):
        # Return empty volume if dir missing
        return (
            np.zeros((1, size[0], size[1]), dtype=np.float32)
            if size
            else np.zeros((1, 512, 512), dtype=np.float32)
        )

    # List and sort files numerically (1.dcm, 2.dcm, ...)
    files = glob.glob(os.path.join(study_dir, "*.dcm"))
    # Sort by integer value of filename
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    volume = []
    for f in files:
        img = load_dicom_array(
            f,
            size=size,
            window_center=Config.WINDOW_CENTER,
            window_width=Config.WINDOW_WIDTH,
        )
        volume.append(img)

    if len(volume) == 0:
        return (
            np.zeros((1, size[0], size[1]), dtype=np.float32)
            if size
            else np.zeros((1, 512, 512), dtype=np.float32)
        )

    volume = np.stack(volume, axis=0)  # (D, H, W)

    # 3. Save to Cache
    # Only cache if we successfully loaded data
    if load_cached_data and len(volume) > 0:
        try:
            np.save(cache_path, volume)
        except Exception as e:
            pass  # Ignore write errors

    return volume


# --------------------------------------------------------------------------
# Segmentation & NIFTI Handling
# --------------------------------------------------------------------------


def load_reoriented_segmentation(path):
    """
    Loads a NIFTI segmentation file and reorients it to match the axial DICOM stack.
    Assumes DICOMs are stacked Inferior-to-Superior or similar, matching the Z-axis
    after canonical reorientation.

    Args:
        path (str): Path to .nii file.

    Returns:
        np.ndarray: 3D mask volume (Depth, Height, Width) or None if failed.
    """
    if not HAS_NIBABEL:
        return None

    try:
        img = nib.load(path)

        # Reorient to Canonical (RAS: Right, Anterior, Superior)
        # This handles the Sagittal -> Axial transformation implicitly if axes are swapped
        ornt = nio.io_orientation(img.affine)
        t_ornt = nio.axcodes2ornt(("R", "A", "S"))
        transform = nio.ornt_transform(ornt, t_ornt)
        img_canonical = img.as_reoriented(transform)

        data = img_canonical.get_fdata()

        # NIFTI Canonical RAS is (Left->Right, Posterior->Anterior, Inferior->Superior)
        # Dimensions are (X, Y, Z).
        # We want (Z, Y, X) to match our DICOM stack (Depth, Height, Width).
        # Also need to check if DICOM Y is Anterior->Posterior or flipped.
        # Standard medical imaging:
        #   Nifti array: [x, y, z]
        #   Desired: [z, y, x]

        data = data.transpose(2, 1, 0)

        # Note: Often Z-axis direction might be flipped relative to DICOM instance numbers.
        # This usually requires checking ImageOrientationPatient from DICOMs.
        # For this implementation, we assume the canonical Z matches the sorted DICOMs.
        # If the segmentation looks flipped in Z during validation, we would add np.flip(data, 0).

        # Rotate to match standard visual orientation if needed (often 90 deg rotation)
        # Standard NIFTI to Numpy often results in rotated images.
        # We return the transposed data; visualization/validation will confirm alignment.
        return data

    except Exception as e:
        # print(f"Error loading NIFTI {path}: {e}")
        return None


def get_roi_center(mask_volume, class_id):
    """
    Calculates the center of mass (Z, Y, X) for a specific class in the segmentation.

    Args:
        mask_volume (np.ndarray): 3D volume (D, H, W).
        class_id (int): The label value to find (1-7).

    Returns:
        tuple: (z, y, x) coordinates normalized to 0-1 range, or None if class not present.
    """
    indices = np.argwhere(mask_volume == class_id)
    if len(indices) == 0:
        return None

    # Calculate mean
    center = indices.mean(axis=0)  # (z, y, x)

    # Normalize
    d, h, w = mask_volume.shape
    norm_center = (center[0] / d, center[1] / h, center[2] / w)

    return norm_center


# --------------------------------------------------------------------------
# Metrics & Loss
# --------------------------------------------------------------------------


class RSNALoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss for RSNA Cervical Spine Fracture Detection.
    """

    def __init__(self, use_competition_weights=True):
        super().__init__()
        # Competition weights: patient_overall is weighted higher.
        # Heuristic: overall=1.0, vertebrae=1/7 (~0.14) or similar ratio.
        # Based on successful solutions:
        #   w_overall = 7/14 = 0.5
        #   w_vert = 1/14 = 0.0714
        if use_competition_weights:
            self.weights = torch.tensor(
                [
                    0.0714,  # C1
                    0.0714,  # C2
                    0.0714,  # C3
                    0.0714,  # C4
                    0.0714,  # C5
                    0.0714,  # C6
                    0.0714,  # C7
                    0.5,  # patient_overall
                ]
            )
        else:
            self.weights = torch.ones(8) / 8.0

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): (Batch, 8) Logits or Probabilities.
            y_true (Tensor): (Batch, 8) Binary targets (0 or 1).
        """
        # Ensure predictions are probabilities
        # If inputs are logits (unbounded), apply sigmoid
        # We assume inputs might be logits for numerical stability with BCEWithLogitsLoss
        # But to strictly implement the formula provided:
        # L = -w * [y * log(p) + (1-y) * log(1-p)]

        # Move weights to device
        weights = self.weights.to(y_pred.device)

        # Use BCEWithLogitsLoss for stability if y_pred are logits
        # If y_pred are already probabilities, we clamp and use raw formula.
        # Assuming logits here as it's standard for PyTorch models.

        loss = F.binary_cross_entropy_with_logits(
            y_pred, y_true.float(), weight=weights, reduction="none"
        )

        # The competition metric is the average of these weighted losses
        return loss.mean()


def weighted_log_loss_numpy(y_pred, y_true):
    """
    Numpy implementation of the metric for validation/inference.
    Expects y_pred to be probabilities (0-1).
    """
    # Weights
    weights = np.array([0.0714] * 7 + [0.5])

    # Clip probabilities to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate binary log loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights
    weighted_loss = loss * weights

    return np.mean(weighted_loss)
