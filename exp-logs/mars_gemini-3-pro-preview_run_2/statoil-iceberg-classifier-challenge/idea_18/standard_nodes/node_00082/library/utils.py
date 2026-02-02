import os
import random
import numpy as np
import torch
import copy
import logging


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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None, level=logging.INFO):
    """
    Creates a logger that logs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
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


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Preserves the best model weights using deepcopy.
    """

    def __init__(self, patience=7, min_delta=0, mode="min", verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when
                the quantity monitored has stopped increasing.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_score, model):

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(current_score, model)
        else:
            if self.mode == "min":
                improved = current_score < (self.best_score - self.min_delta)
            else:
                improved = current_score > (self.best_score + self.min_delta)

            if improved:
                self.best_score = current_score
                self.save_checkpoint(current_score, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model):
        """Saves model state when metric improves."""
        if self.verbose:
            print(f"Metric improved to {score:.6f}. Saving best model state.")
        self.best_state = copy.deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        """Restores the best model weights."""
        if self.best_state is not None:
            if self.verbose:
                print("Restoring best model weights.")
            model.load_state_dict(self.best_state)
        else:
            print("No best state saved to restore.")
