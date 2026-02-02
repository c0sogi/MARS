import os
import random
import numpy as np
import torch
import logging
import pandas as pd
from library.config import SUBMISSION_DIR


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Creates and returns a logger that logs to both console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self, name: str = "Meter", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def rmsle(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Root Mean Squared Logarithmic Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The RMSLE score (scalar tensor).
    """
    # Ensure predictions are non-negative for log
    y_pred = torch.clamp(y_pred, min=0.0)

    # Calculate squared log error
    # log(1 + x) is typically used for RMSLE
    log_true = torch.log1p(y_true)
    log_pred = torch.log1p(y_pred)

    squared_log_error = (log_pred - log_true) ** 2
    mean_squared_log_error = torch.mean(squared_log_error)

    return torch.sqrt(mean_squared_log_error)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    best_filename="model_best.pth.tar",
    save_dir="./working",
):
    """
    Saves the training checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, best_filename)
        torch.save(state, best_filepath)


def save_submission(
    ids, formation_energies, bandgap_energies, filename="submission.csv"
):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: List or array of sample IDs.
        formation_energies: List or array of predicted formation energies.
        bandgap_energies: List or array of predicted bandgap energies.
        filename: Name of the output file.
    """
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_energies,
            "bandgap_energy_ev": bandgap_energies,
        }
    )

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(SUBMISSION_DIR, filename)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
