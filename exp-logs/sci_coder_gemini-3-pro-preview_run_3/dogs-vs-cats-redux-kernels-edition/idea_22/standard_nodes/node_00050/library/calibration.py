import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.utils import get_logger, seed_everything

# Initialize logger
logger = get_logger("calibration")


class TemperatureScaler(nn.Module):
    """
    A PyTorch module that learns a single scalar temperature parameter
    to calibrate model logits.
    """

    def __init__(self, init_temp=1.5):
        super(TemperatureScaler, self).__init__()
        # Initialize temperature to a value > 1 (assuming modern NNs are overconfident)
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        """
        Applies temperature scaling to the logits.
        """
        # Expand temperature to match the size of logits for broadcasting
        # logits shape: (N, 1) or (N,)
        return logits / self.temperature


def optimize_temperature(oof_logits, labels, init_val=1.5, max_iter=50, lr=0.01):
    """
    Optimizes the temperature scalar T to minimize Log Loss on OOF data.

    Args:
        oof_logits (np.ndarray or torch.Tensor): Raw output logits from the model (N, 1) or (N,).
        labels (np.ndarray or torch.Tensor): Ground truth binary labels (N,).
        init_val (float): Initial value for temperature.
        max_iter (int): Maximum iterations for the LBFGS optimizer.
        lr (float): Learning rate.

    Returns:
        float: The optimized temperature value.
    """
    seed_everything(42)

    # Ensure inputs are torch tensors
    if isinstance(oof_logits, np.ndarray):
        oof_logits = torch.from_numpy(oof_logits).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).float()

    # Flatten inputs to ensure shape consistency
    oof_logits = oof_logits.view(-1)
    labels = labels.view(-1)

    # Check for device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof_logits = oof_logits.to(device)
    labels = labels.to(device)

    # Initialize Scaler
    scaler = TemperatureScaler(init_temp=init_val).to(device)

    # Loss function: BCEWithLogitsLoss combines Sigmoid and BCE
    # This is numerically stable and appropriate for binary classification
    nll_criterion = nn.BCEWithLogitsLoss()

    # Optimizer: LBFGS is often used for temperature scaling as it converges
    # quickly for this convex optimization problem.
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        # Calculate scaled logits
        scaled_logits = scaler(oof_logits)
        # Calculate loss
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    # Perform optimization
    initial_loss = nll_criterion(scaler(oof_logits), labels).item()
    optimizer.step(closure)
    final_loss = nll_criterion(scaler(oof_logits), labels).item()

    # Extract temperature
    temp_value = scaler.temperature.item()

    # Safety check: Temperature must be positive.
    # If optimization pushed it <= 0 (unlikely with BCE but possible), reset to 1.0
    if temp_value <= 0:
        logger.warning(
            f"Optimized temperature {temp_value:.4f} <= 0. Resetting to 1.0."
        )
        temp_value = 1.0

    logger.info(
        f"Temperature Optimization: Init Loss={initial_loss:.6f}, Final Loss={final_loss:.6f}, Optimal T={temp_value:.6f}"
    )

    return temp_value


def calibrate_logits(logits, temperature):
    """
    Applies the optimized temperature to logits and returns calibrated probabilities.

    Args:
        logits (np.ndarray or torch.Tensor): Raw logits.
        temperature (float): The scalar temperature value.

    Returns:
        np.ndarray: Calibrated probabilities (0-1).
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()

    # Apply scaling
    scaled_logits = logits / temperature

    # Apply Sigmoid to get probabilities
    probs = torch.sigmoid(scaled_logits)

    return probs.numpy()


class ModelCalibrator:
    """
    Wrapper class to fit and transform using temperature scaling,
    compatible with scikit-learn style pipelines.
    """

    def __init__(self, init_temp=1.5):
        self.init_temp = init_temp
        self.temperature_ = None

    def fit(self, X, y):
        """
        Fits the calibrator.
        X: Raw logits
        y: True labels
        """
        self.temperature_ = optimize_temperature(X, y, init_val=self.init_temp)
        return self

    def transform(self, X):
        """
        Scales logits and returns probabilities.
        """
        if self.temperature_ is None:
            raise ValueError("Calibrator not fitted yet.")
        return calibrate_logits(X, self.temperature_)

    def predict_proba(self, X):
        return self.transform(X)
