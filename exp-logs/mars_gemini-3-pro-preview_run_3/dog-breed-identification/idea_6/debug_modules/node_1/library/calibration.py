import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="calibration")


class TemperatureScaler(nn.Module):
    """
    Implements Temperature Scaling for post-hoc probability calibration.
    Optimizes a single scalar parameter T to minimize NLL on a validation set.
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        # Initialize temperature to 1.5 (common starting heuristic)
        # wrapped in nn.Parameter to allow optimization
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """
        Scales the logits by the optimized temperature.

        Args:
            logits (torch.Tensor): Input logits of shape (N, C).

        Returns:
            torch.Tensor: Scaled logits of shape (N, C).
        """
        # Expand temperature to match logits shape for broadcasting (though scalar div works)
        # We ensure temperature is on the same device as logits
        return logits / self.temperature

    def fit(self, logits, labels):
        """
        Optimizes the temperature parameter using L-BFGS to minimize NLL
        on the provided validation (OOF) data.

        Args:
            logits (torch.Tensor): Validation logits of shape (N, C).
            labels (torch.Tensor): True labels of shape (N).

        Returns:
            self: The fitted scaler instance.
        """
        # Ensure model and data are on the configured device
        self.to(Config.DEVICE)
        logits = logits.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        # Define the objective: Cross Entropy Loss (NLL)
        nll_criterion = nn.CrossEntropyLoss()

        # Calculate NLL before calibration
        with torch.no_grad():
            before_loss = nll_criterion(logits, labels).item()
        logger.info(f"NLL Before Calibration: {before_loss}")

        # L-BFGS optimizer is standard for temperature scaling
        # as it converges very quickly for single-parameter convex problems.
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            # Scale logits using current temperature
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        # Run optimization
        optimizer.step(closure)

        # Calculate NLL after calibration
        with torch.no_grad():
            after_loss = nll_criterion(self.forward(logits), labels).item()

        logger.info(f"NLL After Calibration: {after_loss}")
        logger.info(f"Optimal Temperature: {self.temperature.item()}")

        return self

    def get_probabilities(self, logits):
        """
        Applies temperature scaling and then softmax to get calibrated probabilities.

        Args:
            logits (torch.Tensor): Input logits.

        Returns:
            torch.Tensor: Calibrated probabilities.
        """
        # Ensure logits are on the correct device
        logits = logits.to(Config.DEVICE)

        # Scale logits
        scaled_logits = self.forward(logits)

        # Apply Softmax
        probs = torch.softmax(scaled_logits, dim=1)

        return probs
