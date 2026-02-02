import numpy as np
import torch
from library.config import Config


def get_ash_colors(band11, band14, band15):
    """
    Generates an Ash False Color Composite from GOES-16 infrared bands.
    This composite is used to highlight contrails and volcanic ash, utilizing
    physical optical depth and temperature properties.

    The recipe used is specific to the Contrail Detection task:
    - Red:   Band 15 - Band 14 (Difference)
    - Green: Band 14 - Band 11 (Difference)
    - Blue:  Band 14 (Temperature)

    Args:
        band11 (np.ndarray): Band 11 brightness temperatures (Kelvin).
        band14 (np.ndarray): Band 14 brightness temperatures (Kelvin).
        band15 (np.ndarray): Band 15 brightness temperatures (Kelvin).

    Returns:
        np.ndarray: A 3-channel image array normalized to [0, 1].
                    Shape will match input spatial dimensions with channel dimension last (H, W, 3)
                    or (H, W, T, 3) depending on input.
    """
    # Calculate components
    # Red: Difference between 12.3µm and 11.2µm channels
    r = band15 - band14

    # Green: Difference between 11.2µm and 8.4µm channels
    g = band14 - band11

    # Blue: 11.2µm channel
    b = band14

    # Define normalization bounds (Min, Max)
    # These bounds are empirically derived for this specific satellite product
    r_bounds = (-4.0, 2.0)
    g_bounds = (-4.0, 5.0)
    b_bounds = (243.0, 303.0)

    def normalize(data, bounds):
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    # Normalize components
    r_norm = normalize(r, r_bounds)
    g_norm = normalize(g, g_bounds)
    b_norm = normalize(b, b_bounds)

    # Stack into RGB image
    # Assuming inputs are (H, W) or (H, W, T), we stack along the last new axis
    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)

    # Clip to valid range [0, 1] to handle outliers
    rgb = np.clip(rgb, 0, 1)

    return rgb


def rle_encode(mask):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The metric expects pixels to be numbered from top to bottom, then left to right.
    This corresponds to Column-Major flattening (Fortran style).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of pairs 'start length', or '-' if empty.
    """
    # Flatten in column-major order
    pixels = mask.flatten(order="F")

    # If the mask is completely empty, return '-'
    if not np.any(pixels):
        return "-"

    # Prepend and append 0 to detect transitions at the edges
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run (0->1)
    # runs[1] is the end of the first run (1->0), so length is runs[1] - runs[0]
    # We take starts at even indices and ends at odd indices
    if len(runs) >= 2:
        starts = runs[0::2]
        ends = runs[1::2]
        lengths = ends - starts

        # Interleave starts and lengths
        encoded = []
        for s, l in zip(starts, lengths):
            encoded.append(s)
            encoded.append(l)

        return " ".join(str(x) for x in encoded)

    return "-"


def dice_score_batch(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    """
    Computes the Global Dice Coefficient for a batch of predictions.

    Unlike sample-averaged Dice, this metric treats the entire batch as a single
    volume. This stabilizes the score when many samples in the batch have empty
    ground truth masks, aligning with the "Global Batch Optimization" strategy.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities.
        y_true (torch.Tensor or np.ndarray): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The Global Dice score.
    """
    # Convert numpy arrays to tensors if necessary
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # Binarize predictions
    preds = (y_pred > threshold).float()
    targets = y_true.float()

    # Flatten the entire batch to treat as one global set
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Calculate Intersection and Cardinality
    intersection = (preds_flat * targets_flat).sum()
    cardinality = preds_flat.sum() + targets_flat.sum()

    # Compute Dice
    dice = (2.0 * intersection) / (cardinality + epsilon)

    return dice.item()
