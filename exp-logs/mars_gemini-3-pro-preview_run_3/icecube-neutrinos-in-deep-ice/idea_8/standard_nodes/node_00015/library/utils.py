import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def angles_to_direction(azimuth, zenith):
    """
    Converts azimuth and zenith angles to a 3D unit direction vector (x, y, z).

    Formulas:
        x = cos(azimuth) * sin(zenith)
        y = sin(azimuth) * sin(zenith)
        z = cos(zenith)

    Args:
        azimuth: float, numpy.ndarray, or torch.Tensor. Angle in radians [0, 2*pi].
        zenith: float, numpy.ndarray, or torch.Tensor. Angle in radians [0, pi].

    Returns:
        numpy.ndarray or torch.Tensor: The 3D unit vector(s). Shape (..., 3).
    """
    is_torch = isinstance(azimuth, torch.Tensor) or isinstance(zenith, torch.Tensor)

    if is_torch:
        # Ensure inputs are tensors
        if not isinstance(azimuth, torch.Tensor):
            azimuth = torch.tensor(azimuth)
        if not isinstance(zenith, torch.Tensor):
            zenith = torch.tensor(zenith)

        sin_az = torch.sin(azimuth)
        cos_az = torch.cos(azimuth)
        sin_ze = torch.sin(zenith)
        cos_ze = torch.cos(zenith)

        x = cos_az * sin_ze
        y = sin_az * sin_ze
        z = cos_ze

        return torch.stack([x, y, z], dim=-1)

    else:
        # NumPy implementation
        sin_az = np.sin(azimuth)
        cos_az = np.cos(azimuth)
        sin_ze = np.sin(zenith)
        cos_ze = np.cos(zenith)

        x = cos_az * sin_ze
        y = sin_az * sin_ze
        z = cos_ze

        return np.stack([x, y, z], axis=-1)


def direction_to_angles(vectors):
    """
    Converts 3D unit direction vectors (x, y, z) to azimuth and zenith angles.

    Args:
        vectors: numpy.ndarray or torch.Tensor of shape (..., 3).

    Returns:
        tuple: (azimuth, zenith)
            azimuth: Angle in radians [0, 2*pi].
            zenith: Angle in radians [0, pi].
    """
    is_torch = isinstance(vectors, torch.Tensor)

    if is_torch:
        # Normalize vectors to ensure they are unit length
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]

        # Zenith: z = cos(zenith) => zenith = acos(z)
        # Clip to handle numerical stability issues slightly outside [-1, 1]
        zenith = torch.acos(torch.clamp(z, -1.0, 1.0))

        # Azimuth: atan2(y, x) gives angle in [-pi, pi]
        azimuth = torch.atan2(y, x)

        # Convert range [-pi, pi] to [0, 2*pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)

        return azimuth, zenith

    else:
        # NumPy implementation
        # Normalize vectors
        norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
        # Avoid division by zero
        vectors = vectors / (norm + 1e-8)

        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]

        # Zenith
        zenith = np.arccos(np.clip(z, -1.0, 1.0))

        # Azimuth
        azimuth = np.arctan2(y, x)
        azimuth[azimuth < 0] += 2 * np.pi

        return azimuth, zenith


def angular_error(pred, true_azimuth, true_zenith):
    """
    Computes the angular error (in radians) between predicted direction vectors
    and the ground truth direction defined by azimuth and zenith.

    Args:
        pred: Predicted vectors (numpy array or torch tensor), shape (N, 3).
        true_azimuth: True azimuth angles (radians).
        true_zenith: True zenith angles (radians).

    Returns:
        numpy.ndarray: Array of angular errors in radians.
    """
    # Convert Torch tensors to Numpy if necessary
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true_azimuth, torch.Tensor):
        true_azimuth = true_azimuth.detach().cpu().numpy()
    if isinstance(true_zenith, torch.Tensor):
        true_zenith = true_zenith.detach().cpu().numpy()

    # 1. Convert true angles to unit vectors
    # angles_to_direction handles numpy inputs correctly
    true_vec = angles_to_direction(true_azimuth, true_zenith)

    # 2. Normalize predicted vectors
    norm = np.linalg.norm(pred, axis=1, keepdims=True)
    pred_vec = pred / (norm + 1e-8)

    # 3. Compute Cosine Similarity (Dot Product)
    cos_sim = np.sum(pred_vec * true_vec, axis=1)

    # 4. Clip to [-1, 1] to avoid numerical errors with arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # 5. Compute Angle (Arccos)
    errors = np.arccos(cos_sim)

    return errors
