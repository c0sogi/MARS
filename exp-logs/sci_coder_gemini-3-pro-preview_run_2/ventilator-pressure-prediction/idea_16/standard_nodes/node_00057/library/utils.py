import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (GPU if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that assigns specific weights to the inspiratory and
    expiratory phases of the breath.
    """

    def __init__(self, w_insp: float = 1.0, w_exp: float = 0.1):
        """
        Args:
            w_insp (float): Weight for the inspiratory phase (u_out=0).
            w_exp (float): Weight for the expiratory phase (u_out=1).
        """
        super().__init__()
        self.w_insp = w_insp
        self.w_exp = w_exp
        self.l1 = nn.L1Loss(reduction="none")

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the weighted L1 loss.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth values.
            u_out (torch.Tensor): Control input indicating phase (0 for inspiratory, 1 for expiratory).

        Returns:
            torch.Tensor: The scalar weighted mean loss.
        """
        # Ensure u_out has the same number of dimensions as preds for broadcasting
        # Often u_out is (B, L) while preds is (B, L, 1)
        if u_out.dim() < preds.dim():
            u_out = u_out.unsqueeze(-1)

        # Calculate raw element-wise L1 loss
        loss = self.l1(preds, targets)

        # Generate weight mask:
        # If u_out == 0 (Inspiratory) -> weight = w_insp
        # If u_out == 1 (Expiratory) -> weight = w_exp
        weights = self.w_insp * (1 - u_out) + self.w_exp * u_out

        # Apply weights
        weighted_loss = loss * weights

        # Return the mean over all elements
        return weighted_loss.mean()


def compute_metric(preds: np.ndarray, targets: np.ndarray, u_out: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE) strictly for the inspiratory phase.
    The expiratory phase is not scored.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted pressures.
        targets (np.ndarray or torch.Tensor): Actual pressures.
        u_out (np.ndarray or torch.Tensor): Control signal (0=inspiratory, 1=expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to 1D to handle shape mismatches safely
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Filter for inspiratory phase (u_out == 0)
    # u_out is binary, so we can use boolean indexing
    inspiratory_mask = u_out == 0

    # Safety check to avoid division by zero if no inspiratory phase exists
    if np.sum(inspiratory_mask) == 0:
        return 0.0

    insp_preds = preds[inspiratory_mask]
    insp_targets = targets[inspiratory_mask]

    # Calculate MAE
    mae = np.mean(np.abs(insp_preds - insp_targets))

    return float(mae)
