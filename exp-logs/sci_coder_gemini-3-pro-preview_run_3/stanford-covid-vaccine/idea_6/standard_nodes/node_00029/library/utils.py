import os
import random
import numpy as np
import torch
import torch.nn as nn
import hashlib
import json
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash():
    """
    Generates a unique MD5 hash based on the relevant configuration parameters
    that affect data processing and model architecture. This is used for
    caching preprocessed data.

    Returns:
        str: A hexadecimal hash string.
    """
    # Dictionary of parameters that change the data or model structure
    config_dict = {
        "SEQ_LEN": Config.SEQ_LEN,
        "PRED_LEN": Config.PRED_LEN,
        "INPUT_DIM": Config.INPUT_DIM,
        "OUTPUT_DIM": Config.OUTPUT_DIM,
        "STEM_KERNEL_SIZE": Config.STEM_KERNEL_SIZE,
        "STEM_FILTERS": Config.STEM_FILTERS,
        "RNN_HIDDEN_DIM": Config.RNN_HIDDEN_DIM,
        "RNN_LAYERS": Config.RNN_LAYERS,
        "ATTN_HEADS": Config.ATTN_HEADS,
        "ATTN_LAYERS": Config.ATTN_LAYERS,
        "SCORED_TARGET_INDICES": Config.SCORED_TARGET_INDICES,
    }

    # Serialize to JSON string with sorting to ensure consistency
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss calculates the RMSE for each target column separately and then
    averages them. It handles the slicing of predictions to match the
    ground truth length (seq_scored).
    """

    def __init__(self, select_columns=None):
        """
        Args:
            select_columns (list of int, optional): Indices of columns to include
                in the loss calculation. If None, all columns are used.
                Useful for validation where only specific columns are scored.
        """
        super(MCRMSELoss, self).__init__()
        self.select_columns = select_columns

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, Num_Targets).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, Num_Targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice predictions to match target length
        # Targets are usually length 68 (Config.PRED_LEN)
        # Preds are usually length 107 (Config.SEQ_LEN)
        seq_len_target = targets.shape[1]
        preds_sliced = preds[:, :seq_len_target, :]

        # 2. Select specific columns if requested (e.g., for validation scoring)
        if self.select_columns is not None:
            preds_sliced = preds_sliced[:, :, self.select_columns]
            targets = targets[:, :, self.select_columns]

        # 3. Calculate MSE per column
        # We average over Batch (dim 0) and Sequence (dim 1) first
        mse = torch.mean((preds_sliced - targets) ** 2, dim=(0, 1))

        # 4. Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Calculate Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse)

        return mcrmse
