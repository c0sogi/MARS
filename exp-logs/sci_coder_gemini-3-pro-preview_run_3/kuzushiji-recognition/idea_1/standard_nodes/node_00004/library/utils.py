import os
import random
import torch
import numpy as np
import cv2
from library import config


def setup_directories():
    """
    Creates the necessary directories for working files, cache, and submissions.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size
    and a minimum overlap constraint.

    Args:
        det_size (tuple): (height, width) of the bounding box.
        min_overlap (float): Minimum IoU overlap.

    Returns:
        float: The calculated radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.

    Args:
        shape (tuple): (diameter, diameter) of the kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: The 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap at the specified center.
    Uses element-wise maximum to handle overlapping Gaussians.

    Args:
        heatmap (np.ndarray): The target heatmap array (H, W).
        center (tuple): (x, y) integer coordinates of the center.
        radius (int): Radius of the Gaussian kernel.
        k (float): Scaling factor for the Gaussian peak (default 1).

    Returns:
        np.ndarray: The updated heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def get_3rd_point(a, b):
    """
    Calculates the third point to define an affine transformation.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """
    Calculates the direction vector based on rotation.
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """
    Generates the affine transformation matrix for resizing and cropping.

    Args:
        center (np.ndarray): Center of the source crop (x, y).
        scale (float or np.ndarray): Scale of the source crop (usually max dimension).
        rot (float): Rotation in degrees.
        output_size (tuple): Target size (width, height).
        shift (np.ndarray): Shift applied to the center.
        inv (int): If 1, returns the inverse transform.

    Returns:
        np.ndarray: 2x3 affine transformation matrix.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def affine_transform(pt, t):
    """
    Applies an affine transformation to a 2D point.

    Args:
        pt (np.ndarray or list): Point [x, y].
        t (np.ndarray): 2x3 affine matrix.

    Returns:
        np.ndarray: Transformed point [x, y].
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    """
    Transforms predicted coordinates back to the original image space.

    Args:
        coords (np.ndarray): Predicted coordinates (N, 2) in output space.
        center (np.ndarray): Original image center used for preprocessing.
        scale (float): Original image scale used for preprocessing.
        output_size (tuple): The size of the output feature map (width, height).

    Returns:
        np.ndarray: Transformed coordinates (N, 2) in original image space.
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords
