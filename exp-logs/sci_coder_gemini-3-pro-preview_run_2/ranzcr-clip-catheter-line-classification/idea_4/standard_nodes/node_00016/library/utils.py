import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def get_score(y_true, y_pred):
    """
    Computes the average Area Under the ROC Curve (AUC) across all target labels.
    Handles cases where a specific label might be constant in the provided batch/set.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor), shape (N, C)
        y_pred: Predicted probabilities (numpy array or torch tensor), shape (N, C)

    Returns:
        float: The average AUC score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Check if the column has both classes (0 and 1) to calculate AUC
        if len(np.unique(y_true[:, i])) < 2:
            # If only one class is present, AUC is undefined.
            # We assign 0.5 as a neutral score for this column.
            scores.append(0.5)
        else:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                scores.append(0.5)

    return np.mean(scores)


class ModelEma:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to improve model generalization
    and stability.
    """

    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
            device (torch.device, optional): Device to store the EMA model on.
        """
        # Create a deep copy of the model to store the averaged weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            # Zip state_dict values to handle potential key mismatches
            # (e.g., if the source model is wrapped in DataParallel but EMA module is not)
            # This assumes the architecture and order of parameters are identical.
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """
        Update the EMA parameters using the current model parameters.
        ema_param = decay * ema_param + (1 - decay) * model_param
        """
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        """
        Set the EMA parameters to be exactly the current model parameters.
        """
        self._update(model, update_fn=lambda e, m: m)
