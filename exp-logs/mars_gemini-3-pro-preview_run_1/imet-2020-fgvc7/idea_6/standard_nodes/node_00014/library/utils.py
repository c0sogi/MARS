import torch
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score
from library.config import seed_everything


def calculate_micro_f1(preds, targets, threshold=0.5):
    """
    Calculates the Micro-Averaged F1 score.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities (0-1 range).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels.
        threshold (float): Threshold for classification.

    Returns:
        float: Micro F1 score.
    """
    # Move to CPU and convert to numpy if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions based on threshold
    preds_binary = (preds > threshold).astype(int)
    targets_binary = targets.astype(int)

    return f1_score(targets_binary, preds_binary, average="micro")


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of the model that updates slowly, often leading
    to better generalization and robustness against noisy labels.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor (beta).
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the EMA shadow
        self.ema = deepcopy(model)
        self.ema.eval()

        # Disable gradients for the EMA model to save memory/compute
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current training model.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Iterate over state_dict to handle both parameters and buffers (e.g. BN stats)
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, param in msd.items():
                if name in esd:
                    ema_param = esd[name]

                    # Only apply exponential moving average to floating point tensors
                    if ema_param.dtype in [torch.float16, torch.float32, torch.float64]:
                        # ema_new = decay * ema_old + (1 - decay) * current
                        ema_param.copy_(
                            self.decay * ema_param + (1.0 - self.decay) * param
                        )
                    else:
                        # For integer buffers (e.g., num_batches_tracked), copy directly
                        ema_param.copy_(param)
