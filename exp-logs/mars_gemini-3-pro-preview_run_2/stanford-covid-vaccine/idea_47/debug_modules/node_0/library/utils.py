import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(preds, targets):
    """
    Calculates the MCRMSE loss for training.

    Args:
        preds: (Batch, Seq_Len, 5) Predicted values.
        targets: (Batch, Seq_Len, 5) Ground truth values.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Only evaluate on the scored positions (0 to 67)
    # and the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)

    # Slice to scored sequence length
    preds_scored = preds[:, : Config.SEQ_SCORED, :]
    targets_scored = targets[:, : Config.SEQ_SCORED, :]

    # Slice to scored columns
    preds_filtered = preds_scored[:, :, Config.SCORED_INDICES]
    targets_filtered = targets_scored[:, :, Config.SCORED_INDICES]

    # Calculate MSE per column: Mean over Batch and Sequence
    # dim=0 is Batch, dim=1 is Sequence
    mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

    # Calculate RMSE per column (add epsilon for stability)
    rmse = torch.sqrt(mse + 1e-8)

    # Average RMSE across columns
    loss = torch.mean(rmse)

    return loss


class GlobalMCRMSE:
    """
    Accumulates predictions and targets to calculate the global MCRMSE metric
    over the entire validation set, avoiding batch-averaging bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Store sum of squared errors per column
        # We accumulate on CPU to save GPU memory and avoid device conflicts
        self.sse_per_col = torch.zeros(len(Config.SCORED_INDICES), dtype=torch.float64)
        self.count = 0

    def update(self, preds, targets):
        """
        Updates the running statistics with a new batch of predictions.

        Args:
            preds: (Batch, Seq_Len, 5)
            targets: (Batch, Seq_Len, 5)
        """
        # Move to CPU and detach
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Slice to scored sequence length
        preds_scored = preds[:, : Config.SEQ_SCORED, :]
        targets_scored = targets[:, : Config.SEQ_SCORED, :]

        # Slice to scored columns
        preds_filtered = preds_scored[:, :, Config.SCORED_INDICES]
        targets_filtered = targets_scored[:, :, Config.SCORED_INDICES]

        # Calculate squared errors
        squared_errors = (preds_filtered - targets_filtered) ** 2

        # Sum errors over batch and sequence for each column
        # shape of squared_errors: (Batch, Seq, 3) -> sum over dim 0 and 1 -> (3,)
        batch_sse = torch.sum(squared_errors, dim=(0, 1)).double()

        self.sse_per_col += batch_sse

        # Update count of elements (Batch * Seq_Scored)
        # All columns have the same number of valid elements
        self.count += preds_filtered.shape[0] * preds_filtered.shape[1]

    def compute(self):
        """
        Returns the global MCRMSE score based on accumulated data.

        Returns:
            float: The MCRMSE score.
        """
        if self.count == 0:
            return 0.0

        # MSE per column
        mse_per_col = self.sse_per_col / self.count

        # RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Mean Columnwise RMSE
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()


def format_submission(test_ids, test_preds, save_path=Config.SUBMISSION_PATH):
    """
    Formats the predictions into the competition submission format and saves to CSV.

    Args:
        test_ids: List of sequence IDs.
        test_preds: Numpy array of shape (Num_Samples, Seq_Len, 5).
        save_path: Path to save the CSV.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    id_seqpos_list = []
    preds_list = []

    # Iterate over each sample
    for i, sample_id in enumerate(test_ids):
        # Get predictions for this sample: (107, 5)
        sample_preds = test_preds[i]

        # Generate id_seqpos keys for all 107 positions
        for seqpos in range(Config.SEQ_LENGTH):
            id_seqpos_list.append(f"{sample_id}_{seqpos}")
            preds_list.append(sample_preds[seqpos])

    # Create DataFrame
    cols = (
        Config.TARGET_COLS
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    preds_array = np.array(preds_list)

    submission_df = pd.DataFrame(preds_array, columns=cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save
    submission_df.to_csv(save_path, index=False)
