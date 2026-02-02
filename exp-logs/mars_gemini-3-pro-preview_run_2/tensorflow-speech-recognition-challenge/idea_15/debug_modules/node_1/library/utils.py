import os
import random
import copy
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEMA:
    """
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device if device else Config.DEVICE

        # Create a shadow copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.to(self.device)

        # Disable gradients for the shadow model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model parameters.
        """
        with torch.no_grad():
            # Update parameters
            msd = model.state_dict()
            esd = self.ema_model.state_dict()

            for name, param in msd.items():
                if name in esd:
                    # Determine the type of the parameter/buffer
                    if param.dtype.is_floating_point:
                        # EMA update for floating point parameters/buffers
                        esd[name].copy_(
                            self.decay * esd[name] + (1.0 - self.decay) * param
                        )
                    else:
                        # Direct copy for integer buffers (e.g. num_batches_tracked)
                        esd[name].copy_(param)


def save_checkpoint(model, optimizer, epoch, score, filepath):
    """
    Saves the model checkpoint including optimizer state and metrics.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "score": score,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)


def calculate_accuracy(outputs, targets):
    """
    Calculates multiclass accuracy.

    Args:
        outputs: Logits or probabilities from the model (Batch, Num_Classes)
        targets: Ground truth labels (Batch,)

    Returns:
        accuracy: float
    """
    with torch.no_grad():
        _, predictions = torch.max(outputs, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        if total == 0:
            return 0.0
        return correct / total
