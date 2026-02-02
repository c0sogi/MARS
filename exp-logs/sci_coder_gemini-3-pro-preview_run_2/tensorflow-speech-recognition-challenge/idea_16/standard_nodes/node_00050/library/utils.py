import os
import torch
import logging
from copy import deepcopy
from library.config import Config, set_seed

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, metrics, filename="checkpoint.pth"):
    """
    Saves the model checkpoint, optimizer state, and metrics to the working directory.

    Args:
        model (torch.nn.Module or ModelEMA): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        metrics (dict): Validation metrics.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # If model is wrapped in ModelEMA, save the shadow model's state_dict
    if isinstance(model, ModelEMA):
        model_state = model.ema_model.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "metrics": metrics,
    }

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    model, optimizer=None, filename="checkpoint.pth", device=Config.DEVICE
):
    """
    Loads a checkpoint from the working directory.

    Args:
        model (torch.nn.Module or ModelEMA): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Name of the checkpoint file.
        device (str): Device to map the checkpoint to.

    Returns:
        tuple: (epoch, metrics) if loaded, else (None, None).
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        logger.info(f"No checkpoint found at {filepath}. Starting fresh.")
        return None, None

    logger.info(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    if isinstance(model, ModelEMA):
        model.ema_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch")
    metrics = checkpoint.get("metrics")

    logger.info(f"Checkpoint loaded. Resuming from epoch {epoch}.")
    return epoch, metrics


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    Maintains a shadow copy of the model that is updated using a decay factor.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=Config.DEVICE):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
            device (str): Device to store the shadow model on.
        """
        self.decay = decay
        self.device = device
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.to(self.device)

        # Ensure EMA parameters do not require gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters based on the current model.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for name, ema_v in self.ema_model.state_dict().items():
                if name in msd:
                    model_v = msd[name].to(self.device)

                    if ema_v.dtype.is_floating_point:
                        # Update floating point parameters with decay
                        ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
                    else:
                        # Directly copy integer buffers (e.g., num_batches_tracked)
                        ema_v.copy_(model_v)

    def to(self, device):
        self.ema_model.to(device)
        self.device = device
        return self


def log_metrics(metrics):
    """
    Logs validation metrics with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    if not metrics:
        return

    # Print full precision as requested
    msg_parts = [f"{k}: {v}" for k, v in metrics.items()]
    logger.info("Validation Metrics: " + " | ".join(msg_parts))
