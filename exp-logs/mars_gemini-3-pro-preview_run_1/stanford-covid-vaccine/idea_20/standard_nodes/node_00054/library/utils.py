import os
import random
import numpy as np
import torch
import logging
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", float("inf"))


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This implementation follows the "Metric Correction" strategy:
    1. Compute MSE for each of the 3 scored columns independently.
    2. Compute RMSE for each column.
    3. Average the 3 RMSE values.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Scored, n_targets).
        y_pred (torch.Tensor): Predicted tensor of shape (Batch, Seq_Scored, n_targets).

    Returns:
        torch.Tensor: Scalar tensor representing the MCRMSE.
    """
    # Calculate MSE per column (averaging over batch and sequence length dimensions)
    # Dimensions: 0=Batch, 1=Sequence Position, 2=Target Column
    mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Average RMSE across the 3 columns
    mcrmse = torch.mean(rmse)

    return mcrmse


def generate_submission_file(ids, sequences, predictions, output_path):
    """
    Generates the submission CSV file in the required format.

    Args:
        ids (list): List of sample IDs.
        sequences (list): List of sequences (used for length verification).
        predictions (np.ndarray): Array of shape (N_samples, Seq_Len, 3).
                                  Columns correspond to [reactivity, deg_Mg_pH10, deg_Mg_50C].
        output_path (str): Path to save the submission CSV.
    """
    # Submission columns order as per requirements
    sub_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Prepare lists for dataframe construction
    id_seqpos_list = []
    data_map = {col: [] for col in sub_cols}

    # Iterate over samples
    for i, sample_id in enumerate(ids):
        seq_len = len(sequences[i])

        # Predictions for this sample: Shape (107, 3)
        pred_sample = predictions[i]

        for pos in range(seq_len):
            id_seqpos = f"{sample_id}_{pos}"
            id_seqpos_list.append(id_seqpos)

            # Map predictions to columns
            # Model Output Index 0 -> reactivity
            # Model Output Index 1 -> deg_Mg_pH10
            # Model Output Index 2 -> deg_Mg_50C

            data_map["reactivity"].append(pred_sample[pos, 0])
            data_map["deg_Mg_pH10"].append(pred_sample[pos, 1])
            data_map["deg_pH10"].append(0.0)  # Unscored, fill with 0
            data_map["deg_Mg_50C"].append(pred_sample[pos, 2])
            data_map["deg_50C"].append(0.0)  # Unscored, fill with 0

    # Create DataFrame
    df_sub = pd.DataFrame()
    df_sub["id_seqpos"] = id_seqpos_list
    for col in sub_cols:
        df_sub[col] = data_map[col]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
