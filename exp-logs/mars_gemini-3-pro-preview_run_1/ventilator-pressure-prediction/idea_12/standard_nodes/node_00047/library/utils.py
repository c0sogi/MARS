import torch
import numpy as np
from library.config import Config, seed_everything


def get_device():
    """
    Returns the PyTorch device configured in Config.
    """
    return torch.device(Config.DEVICE)


def compute_mae(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.

    The metric is defined as the MAE between predictions and targets,
    considering only time steps where the expiratory valve is closed (u_out == 0).

    Args:
        preds: Predicted pressure values (torch.Tensor or np.ndarray).
        targets: Actual pressure values (torch.Tensor or np.ndarray).
        u_out: Control input for expiratory valve (torch.Tensor or np.ndarray).
               0 indicates inspiratory phase, 1 indicates expiratory phase.

    Returns:
        float: The calculated MAE for the inspiratory phase.
    """
    # Determine if inputs are PyTorch tensors
    is_tensor = torch.is_tensor(preds)

    if is_tensor:
        # Ensure all inputs are on the same device and are tensors
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, device=preds.device)
        if not torch.is_tensor(u_out):
            u_out = torch.tensor(u_out, device=preds.device)

        # Create boolean mask for inspiratory phase (u_out == 0)
        # u_out might be float or int, so we compare strictly or close to 0
        mask = u_out == 0

        # Apply mask
        masked_preds = preds[mask]
        masked_targets = targets[mask]

        # Compute MAE
        mae = torch.abs(masked_preds - masked_targets).mean()

        return mae.item()

    else:
        # Assume numpy arrays
        preds = np.array(preds)
        targets = np.array(targets)
        u_out = np.array(u_out)

        # Create boolean mask
        mask = u_out == 0

        # Apply mask
        masked_preds = preds[mask]
        masked_targets = targets[mask]

        # Compute MAE
        mae = np.mean(np.abs(masked_preds - masked_targets))

        return float(mae)
