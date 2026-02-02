import torch
import numpy as np
from library.config import Config


def get_sinusoidal_encoding_table(n_position, d_hid):
    """
    Generates a sinusoidal positional encoding table.

    This function creates a lookup table of fixed sinusoidal embeddings,
    commonly used for positional encoding in Transformers or for encoding
    geometric distances in this RNA degradation task.

    Args:
        n_position (int): The number of unique positions (or distance buckets) to encode.
                          This determines the number of rows in the lookup table.
        d_hid (int): The dimension of the embedding vectors (hidden dimension).

    Returns:
        torch.FloatTensor: A tensor of shape (n_position, d_hid) containing the
                           sinusoidal encodings.
    """

    def get_position_angle_vec(position):
        # Calculate the angle for each dimension index j
        # angle = pos / 10000^(2 * (j // 2) / d_hid)
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    # Create the table using the helper function for each position
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    # Apply sine to even indices (2i)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])

    # Apply cosine to odd indices (2i+1)
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # Convert to a FloatTensor
    return torch.FloatTensor(sinusoid_table)


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric is the primary evaluation criteria for the task. It is calculated by:
    1. Slicing the predictions and ground truth to the scored sequence length (Config.PRED_LEN).
    2. Computing the Root Mean Squared Error (RMSE) for each target column independently.
    3. Averaging the RMSE values across all target columns.

    This implementation assumes y_true and y_pred are aligned in terms of target columns
    (e.g., both contain [reactivity, deg_Mg_pH10, deg_Mg_50C]).

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Len, Num_Targets).
        y_pred (torch.Tensor): Predicted tensor of shape (Batch, Seq_Len, Num_Targets).

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE value.
    """
    # Ensure inputs are on the same device
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Slice to the scored length (first 68 positions as defined in Config)
    # Shape becomes: (Batch, 68, Num_Targets)
    y_true_scored = y_true[:, : Config.PRED_LEN, :]
    y_pred_scored = y_pred[:, : Config.PRED_LEN, :]

    # Calculate Mean Squared Error (MSE) for each column
    # We average over the batch (dim 0) and sequence (dim 1) dimensions,
    # but keep the column dimension (dim 2) separate initially.
    mse = torch.mean((y_true_scored - y_pred_scored) ** 2, dim=(0, 1))

    # Calculate Root Mean Squared Error (RMSE) for each column
    rmse = torch.sqrt(mse)

    # Calculate the Mean of the RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse
