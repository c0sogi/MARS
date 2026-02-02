import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.

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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score.
    Robustly handles cases where a class might not be present (or fully present)
    in the current batch/split by skipping that class in the average.

    Args:
        y_true: Ground truth labels (numpy array or tensor).
        y_pred: Predicted probabilities (numpy array or tensor).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Only calculate AUC if the class has both 0 and 1 labels in the true set
        # This prevents ValueError from sklearn when a class is constant in the batch
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                pass

    if not auc_scores:
        # If no classes are valid (e.g., extremely small batch with constant labels), return 0.5
        return 0.5

    return np.mean(auc_scores)


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Maintains a shadow model that updates slowly, stabilizing training
    on small datasets and improving generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=None):
        """
        Args:
            model: The model to track.
            decay: The decay rate for EMA (default from Config).
            device: Device to store the shadow model on.
        """
        self.decay = decay
        self.device = device if device else Config.DEVICE
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.shadow.to(self.device)

        # Disable gradients for shadow model parameters to save memory/compute
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model weights using the current model weights.
        Formula: shadow = decay * shadow + (1 - decay) * model

        Args:
            model: The current active training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            ssd = self.shadow.state_dict()

            for key in msd:
                if key in ssd:
                    # Move model parameter to the correct device for calculation
                    model_param = msd[key].to(self.device)
                    shadow_param = ssd[key]

                    # Update float parameters using EMA formula
                    if model_param.dtype in [
                        torch.float16,
                        torch.float32,
                        torch.float64,
                    ]:
                        shadow_param.mul_(self.decay).add_(
                            model_param, alpha=(1.0 - self.decay)
                        )
                    # Directly copy integer buffers (e.g., num_batches_tracked in BatchNorm)
                    else:
                        shadow_param.copy_(model_param)
