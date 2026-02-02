import numpy as np
import cv2


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The format consists of space-delimited pairs of values: start position and run length.
    Pixels are 1-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (numpy.ndarray): Binary mask of shape (H, W), where 1 indicates salt
                              and 0 indicates sediment.

    Returns:
        str: RLE string (e.g., '1 3 10 5'). Returns an empty string for empty masks.
    """
    # Flatten in column-major order (Fortran-style) as required by the task
    pixels = mask.flatten(order="F")

    # Pad with 0 at start and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end_pos - start_pos)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        rle_str (str): RLE string. Can be empty or NaN.
        shape (tuple): The shape of the output mask (H, W). Defaults to (101, 101).

    Returns:
        numpy.ndarray: Binary mask of shape `shape` with dtype uint8.
    """
    if not isinstance(rle_str, str) or not rle_str:
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-indexed to 0-indexed
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to 2D using column-major order
    return img.reshape(shape, order="F")


def pad_image(image, target_size=(128, 128)):
    """
    Pads an image from its original size (e.g., 101x101) to a target size (e.g., 128x128)
    using reflection padding. This is necessary for network architectures that require
    dimensions divisible by powers of 2 (e.g., 32).

    Args:
        image (numpy.ndarray): Input image or mask. Shape (H, W) or (H, W, C).
        target_size (tuple): Target spatial dimensions (H, W).

    Returns:
        numpy.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    diff_h = target_h - h
    diff_w = target_w - w

    if diff_h < 0 or diff_w < 0:
        raise ValueError("Target size must be larger than original image size.")

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # cv2.copyMakeBorder handles both 2D and 3D arrays correctly
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
    )

    # Ensure channel dimension is preserved if it existed but was 1 (cv2 might drop it)
    if len(image.shape) == 3 and len(padded.shape) == 2:
        padded = padded[:, :, np.newaxis]

    return padded


def unpad_image(image, original_size=(101, 101)):
    """
    Crops a padded image back to its original dimensions (center crop).

    Args:
        image (numpy.ndarray): Padded image.
        original_size (tuple): Desired output dimensions (H, W).

    Returns:
        numpy.ndarray: Unpadded image.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = original_size

    diff_h = h - orig_h
    diff_w = w - orig_w

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    return image[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds ranging from 0.5 to 0.95
    with a step size of 0.05.

    Args:
        predict (numpy.ndarray): Predicted output. Can be probabilities or binary masks.
                                 Shape (N, H, W).
        truth (numpy.ndarray): Ground truth binary masks. Shape (N, H, W).
        threshold (float): Threshold to binarize `predict` if it contains probabilities.
                           Defaults to 0.5.

    Returns:
        float: The average precision score averaged over the batch.
    """
    N = len(predict)
    batch_scores = []

    # Binarize predictions
    predict_binary = (predict > threshold).astype(np.uint8)
    truth_binary = (truth > 0.5).astype(np.uint8)

    # Define IoU thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    for i in range(N):
        p = predict_binary[i]
        t = truth_binary[i]

        sum_p = np.sum(p)
        sum_t = np.sum(t)

        # Calculate IoU
        if sum_p == 0 and sum_t == 0:
            iou = 1.0
        elif sum_p == 0 or sum_t == 0:
            iou = 0.0
        else:
            intersection = np.sum((p * t) > 0)
            union = np.sum((p + t) > 0)
            iou = intersection / union

        # Calculate precision for this image:
        # At each threshold t, precision is 1 if IoU > t, else 0.
        # Average precision is the mean of these binary scores.
        # Using > (strictly greater) as per typical competition metric implementations,
        # though some definitions use >=. The prompt says "greater than 0.5".
        matches = iou > iou_thresholds
        image_score = np.mean(matches)
        batch_scores.append(image_score)

    return np.mean(batch_scores)
