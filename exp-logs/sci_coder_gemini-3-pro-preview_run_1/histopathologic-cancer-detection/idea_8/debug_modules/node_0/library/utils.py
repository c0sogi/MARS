import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
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

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_scores):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted probabilities for the positive class.

    Returns:
        float: The AUC score.
    """
    # Detach tensors if necessary and convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5


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


def save_checkpoint(state, filename):
    """
    Saves the model state to a file.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


class EarlyStopping:
    """
    Implements early stopping logic to stop training when validation metric stops improving.
    Also handles saving the best model checkpoint.
    """

    def __init__(
        self, patience=6, mode="max", delta=0.0, verbose=False, path="checkpoint.pth"
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.val_score_max = -np.inf

        if self.mode == "min":
            self.monitor_op = np.less
        elif self.mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError(
                f"EarlyStopping mode {mode} is unknown, use 'min' or 'max'."
            )

    def __call__(self, score, model, optimizer=None, epoch=None):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, epoch)
        else:
            # Check if score improved based on mode and delta
            if self.mode == "min":
                improved = score < (self.best_score - self.delta)
            else:
                improved = score > (self.best_score + self.delta)

            if improved:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, epoch)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model, optimizer, epoch):
        """Saves model when validation metric decreases."""
        if self.verbose:
            print(f"Validation metric improved. Saving model to {self.path}")

        state = {"model_state_dict": model.state_dict(), "score": score, "epoch": epoch}
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()

        save_checkpoint(state, self.path)
