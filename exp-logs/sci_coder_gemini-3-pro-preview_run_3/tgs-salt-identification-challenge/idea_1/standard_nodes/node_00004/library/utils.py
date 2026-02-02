import numpy as np


def rle_encode(img):
    """
    Convert a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask where 1 indicates the object and 0 background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-wise (Fortran-style) as per competition spec
    # 1 is pixel (1,1), 2 is pixel (2,1), etc.
    pixels = img.flatten(order="F")

    # Pad with 0s to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    # The runs array now looks like [start1, end1, start2, end2, ...]
    # We want to convert end indices to lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decode a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask of the specified shape.
    """
    if mask_rle is None or str(mask_rle) == "nan" or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-based indexing to 0-based
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (Fortran-style)
    return img.reshape(shape, order="F")


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculate the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (np.ndarray): Predicted binary masks, shape (N, H, W).
                              Values should be 0 or 1.
        truth (np.ndarray): Ground truth binary masks, shape (N, H, W).
                            Values should be 0 or 1.
        threshold (float): Unused, kept for compatibility with standard metric signatures.

    Returns:
        float: The mean average precision score.
    """
    N = len(predict)
    if N == 0:
        return 0.0

    # Flatten spatial dimensions for vectorized IoU calculation
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    # Calculate Intersection and Union
    intersection = (predict * truth).sum(axis=1)
    union = (predict + truth).sum(axis=1) - intersection

    # Calculate IoU
    # Handle empty union (both pred and truth are empty) -> IoU = 1
    iou = np.ones(N)
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    # Define thresholds: 0.5, 0.55, ..., 0.95
    # Using explicit list to avoid floating point precision issues with arange
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    # Compare IoU to thresholds
    # Broadcasting: (N, 1) > (1, 10) -> (N, 10) boolean matrix
    matches = iou[:, None] > thresholds[None, :]

    # Average over thresholds for each image -> Precision per image
    # If IoU > t, precision is 1, else 0. The mean is the average precision.
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return image_scores.mean()


class MinMaxNormalizer:
    """
    Utility class to normalize values to [0, 1] range.
    """

    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, values):
        """Compute min and max from the data."""
        self.min_val = np.min(values)
        self.max_val = np.max(values)

    def transform(self, values):
        """Scale values to [0, 1] using fitted min and max."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before transform.")

        denom = self.max_val - self.min_val
        if denom == 0:
            return np.zeros_like(values, dtype=float)

        return (values - self.min_val) / denom

    def fit_transform(self, values):
        """Fit and transform in one step."""
        self.fit(values)
        return self.transform(values)
