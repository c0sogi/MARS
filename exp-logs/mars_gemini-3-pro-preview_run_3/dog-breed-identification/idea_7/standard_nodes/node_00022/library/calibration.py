import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config


class TemperatureScaler(nn.Module):
    """
    Implements Temperature Scaling for Post-Hoc Calibration of neural networks.
    Optimizes a single scalar parameter T to minimize NLL on a validation set.
    """

    def __init__(self, device=Config.DEVICE):
        super(TemperatureScaler, self).__init__()
        self.device = device
        # Initialize temperature to 1.5 as a starting guess.
        # It is a learnable parameter.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """
        Applies temperature scaling to the logits.

        Args:
            logits (torch.Tensor): Input logits of shape (N, C).

        Returns:
            torch.Tensor: Scaled logits (logits / T).
        """
        # Expand temperature to match logits shape for broadcasting if necessary,
        # though scalar division handles this automatically in PyTorch.
        return logits / self.temperature

    def fit(self, logits, labels):
        """
        Tunes the temperature parameter using LBFGS to minimize Negative Log Likelihood (NLL)
        on the provided validation set.

        Args:
            logits (torch.Tensor): Validation logits of shape (N, C).
            labels (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            self: The fitted scaler instance.
        """
        self.to(self.device)
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        # Define the loss function (NLL)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        # Calculate and print NLL before calibration
        with torch.no_grad():
            before_nll = nll_criterion(logits, labels).item()
        print(f"NLL Before Calibration: {before_nll}")

        # LBFGS optimizer is the standard choice for Temperature Scaling
        # as it converges quickly for a single parameter optimization.
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Calculate and print NLL after calibration
        with torch.no_grad():
            after_nll = nll_criterion(self.forward(logits), labels).item()

        print(f"NLL After Calibration: {after_nll}")
        print(f"Optimal Temperature: {self.temperature.item()}")

        return self

    def predict_proba(self, logits):
        """
        Scales logits and applies Softmax to return calibrated probabilities.

        Args:
            logits (torch.Tensor): Input logits.

        Returns:
            torch.Tensor: Calibrated probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = logits.to(self.device)
            scaled_logits = self.forward(logits)
            probs = torch.softmax(scaled_logits, dim=1)
        return probs

    def get_temperature(self):
        """
        Returns the optimized temperature value.
        """
        return self.temperature.item()
