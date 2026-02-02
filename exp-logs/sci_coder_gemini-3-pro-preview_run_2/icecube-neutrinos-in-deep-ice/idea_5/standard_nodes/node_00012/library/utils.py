import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates to cartesian coordinates.

    Args:
        azimuth (np.array or torch.Tensor): Azimuth angle in radians [0, 2*pi].
        zenith (np.array or torch.Tensor): Zenith angle in radians [0, pi].

    Returns:
        x, y, z: Normalized cartesian coordinates.
    """
    # Ensure inputs are of the same type (numpy or torch)
    if isinstance(azimuth, torch.Tensor):
        sin_zen = torch.sin(zenith)
        x = torch.cos(azimuth) * sin_zen
        y = torch.sin(azimuth) * sin_zen
        z = torch.cos(zenith)
    else:
        sin_zen = np.sin(zenith)
        x = np.cos(azimuth) * sin_zen
        y = np.sin(azimuth) * sin_zen
        z = np.cos(zenith)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts cartesian coordinates to spherical coordinates.

    Args:
        x, y, z (np.array or torch.Tensor): Cartesian coordinates.

    Returns:
        azimuth, zenith: Angles in radians.
    """
    if isinstance(x, torch.Tensor):
        # Normalize vector just in case
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / (norm + 1e-8), y / (norm + 1e-8), z / (norm + 1e-8)

        zenith = torch.acos(torch.clamp(z, -1.0, 1.0))
        azimuth = torch.atan2(y, x)
        # Convert azimuth from [-pi, pi] to [0, 2*pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * math.pi, azimuth)
    else:
        # Normalize vector
        norm = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / (norm + 1e-8), y / (norm + 1e-8), z / (norm + 1e-8)

        zenith = np.arccos(np.clip(z, -1.0, 1.0))
        azimuth = np.arctan2(y, x)
        # Convert azimuth from [-pi, pi] to [0, 2*pi]
        azimuth = np.where(azimuth < 0, azimuth + 2 * math.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(azimuth_true, zenith_true, azimuth_pred, zenith_pred):
    """
    Calculates the mean angular error between predicted and true directions.

    Args:
        azimuth_true, zenith_true: Ground truth angles (numpy arrays).
        azimuth_pred, zenith_pred: Predicted angles (numpy arrays).

    Returns:
        float: Mean angular error in radians.
    """
    # Convert to cartesian vectors
    x_true, y_true, z_true = spherical_to_cartesian(azimuth_true, zenith_true)
    x_pred, y_pred, z_pred = spherical_to_cartesian(azimuth_pred, zenith_pred)

    # Compute dot product: u . v
    # Since vectors are normalized, dot product is cosine of angle
    dot_product = x_true * x_pred + y_true * y_pred + z_true * z_pred

    # Clip for numerical stability (acos domain is [-1, 1])
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle
    angular_errors = np.arccos(dot_product)

    return np.mean(angular_errors)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule (default 0.5 for half-cosine).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
