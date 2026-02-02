import numpy as np
import torch
import torch.nn as nn
import os
from library.config import Config


def align_to_canonical_view(skeleton):
    """
    Aligns the skeleton sequence to a canonical view frame-by-frame to ensure
    user independence.

    Transformation:
    1. Translates the skeleton so HipCenter (Joint 0) is at the origin.
    2. Rotates the skeleton around the Y-axis so that the vector connecting
       HipLeft (Joint 12) to HipRight (Joint 16) is parallel to the global X-axis.

    Args:
        skeleton (np.ndarray): Raw skeleton data of shape (NumFrames, 20, 3).

    Returns:
        np.ndarray: Aligned skeleton data of shape (NumFrames, 20, 3).
    """
    # Joint Indices based on Kinect v1 standard / Dataset Description
    HIP_CENTER_IDX = 0
    HIP_LEFT_IDX = 12
    HIP_RIGHT_IDX = 16

    # Ensure we don't modify the original array
    aligned = np.copy(skeleton)

    # 1. Translation: Center at HipCenter
    # Shape: (T, 1, 3)
    offsets = aligned[:, HIP_CENTER_IDX : HIP_CENTER_IDX + 1, :]
    aligned = aligned - offsets

    # 2. Rotation: Align Hips to X-axis
    # Vector from Left Hip to Right Hip
    # Shape: (T, 3)
    hip_vec = aligned[:, HIP_RIGHT_IDX, :] - aligned[:, HIP_LEFT_IDX, :]

    # Calculate angle in the XZ plane (Y is up)
    # We want to rotate this vector to align with (1, 0, 0)
    # Current angle theta = arctan2(z, x)
    # Required rotation = -theta
    angles = np.arctan2(hip_vec[:, 2], hip_vec[:, 0])

    # Compute rotation terms
    # We rotate by -angles. cos(-a) = cos(a), sin(-a) = -sin(a)
    c = np.cos(-angles)
    s = np.sin(-angles)

    # Construct Rotation Matrices for rotation around Y-axis
    # R_y = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    # Note: The 's' in the matrix corresponds to sin(rotation_angle).
    # Since rotation_angle is -theta, sin(-theta) is -sin(theta).
    # Let s_val = sin(-theta).

    s_val = np.sin(-angles)
    c_val = np.cos(-angles)

    zeros = np.zeros_like(c_val)
    ones = np.ones_like(c_val)

    # Build (T, 3, 3) rotation matrices
    # Row 0: [c, 0, s] -> [c_val, 0, s_val] (Standard Ry definition)
    # Row 1: [0, 1, 0]
    # Row 2: [-s, 0, c] -> [-s_val, 0, c_val]

    # Stack columns to form rows, then stack rows
    R = np.stack(
        [
            np.stack([c_val, zeros, s_val], axis=1),
            np.stack([zeros, ones, zeros], axis=1),
            np.stack([-s_val, zeros, c_val], axis=1),
        ],
        axis=1,
    )

    # Apply Rotation: v' = R v
    # einsum: t=time, i=row(out), j=col(in), p=joint
    aligned = np.einsum("tij,tpj->tpi", R, aligned)

    return aligned


def compute_kinematics(skeleton):
    """
    Computes first (Velocity) and second (Acceleration) derivatives of the
    skeleton positions and concatenates them.

    Args:
        skeleton (np.ndarray): Position data of shape (NumFrames, 20, 3).

    Returns:
        np.ndarray: Augmented features of shape (NumFrames, 20, 9).
                    Format: [Position, Velocity, Acceleration]
    """
    # Compute Velocity (1st derivative) using central differences
    # np.gradient automatically handles boundaries
    velocity = np.gradient(skeleton, axis=0)

    # Compute Acceleration (2nd derivative)
    acceleration = np.gradient(velocity, axis=0)

    # Concatenate along the coordinate dimension (last axis)
    return np.concatenate([skeleton, velocity, acceleration], axis=2)


class TruncatedMSELoss(nn.Module):
    """
    Truncated Mean Squared Error Loss applied to log-probabilities.
    Used for temporal smoothing to penalize rapid fluctuations in predictions
    without being overly sensitive to genuine sharp transitions.
    """

    def __init__(self, threshold=1.0):
        """
        Args:
            threshold (float): Maximum squared error value to clip the loss to.
        """
        super(TruncatedMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Log-probabilities of shape (Batch, Classes, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate difference between adjacent frames
        # diff[t] = log_probs[t+1] - log_probs[t]
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # We want the difference to be 0 (smoothness)
        target = torch.zeros_like(diff)

        # Calculate squared error
        loss = self.mse(diff, target)

        # Truncate the loss to prevent outliers from dominating gradients
        loss = torch.clamp(loss, max=self.threshold**2)

        return loss.mean()


def rle_decode(predictions):
    """
    Decodes frame-wise class predictions into an ordered list of gesture IDs.
    Collapses consecutive duplicates and removes the background class.

    Args:
        predictions (list or np.ndarray): Sequence of frame-wise class IDs.

    Returns:
        list: Ordered list of gesture IDs (integers).
    """
    if len(predictions) == 0:
        return []

    # Run-Length Encoding: Collapse consecutive duplicates
    collapsed = [predictions[0]]
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i - 1]:
            collapsed.append(predictions[i])

    # Filter out Background Class (0)
    # The challenge requires only the list of 20 interest gestures
    final_gestures = [int(g) for g in collapsed if g != Config.BACKGROUND_CLASS_ID]

    return final_gestures


def save_submission(predictions_dict, output_path):
    """
    Saves the predictions to a CSV file in the format required by the challenge.

    Format:
    SessionID,Label1,Label2,Label3

    Args:
        predictions_dict (dict): Dictionary mapping SessionID (str) to list of gesture IDs.
        output_path (str): Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for session_id, labels in predictions_dict.items():
            # Convert labels to string
            label_str = ",".join(map(str, labels))
            f.write(f"{session_id},{label_str}\n")
