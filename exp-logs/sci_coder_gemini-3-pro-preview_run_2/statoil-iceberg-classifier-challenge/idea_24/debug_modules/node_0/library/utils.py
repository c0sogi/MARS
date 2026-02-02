import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in CuDNN.

    Args:
        seed (int): The seed value to use.
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


def calculate_metrics(y_true, y_pred_logits):
    """
    Calculates Log Loss and Accuracy from ground truth and model logits.

    Args:
        y_true (np.array): Ground truth binary labels (0 or 1).
        y_pred_logits (np.array): Raw output logits from the model.

    Returns:
        dict: A dictionary containing 'log_loss' and 'accuracy'.
    """
    # Convert logits to probabilities using Sigmoid
    y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred_logits))

    # Calculate Log Loss (scikit-learn handles epsilon clipping internally, but we pass probs)
    # labels=[0, 1] ensures correct handling even if a batch misses a class
    loss = log_loss(y_true, y_pred_probs, labels=[0, 1])

    # Calculate Accuracy (Threshold = 0.5)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)

    return {"log_loss": loss, "accuracy": acc}


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Strictly preserves the best model weights using copy.deepcopy.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): Trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        """
        Call method to update early stopping status.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        Stores the best state_dict in memory and saves to disk.
        """
        if self.verbose:
            # Print full precision as requested
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Strictly preserve the best weights in memory using deepcopy
        self.best_model_wts = copy.deepcopy(model.state_dict())

        # Save to disk
        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss

    def load_best_weights(self, model):
        """
        Restores the best weights to the provided model instance.
        """
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
