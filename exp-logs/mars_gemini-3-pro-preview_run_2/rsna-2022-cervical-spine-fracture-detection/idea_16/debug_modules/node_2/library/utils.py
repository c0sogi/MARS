import torch
import torch.nn.functional as F
from library.config import Config, seed_everything


def get_class_weights(device):
    """
    Retrieves the class weights from Config and moves them to the specified device.

    Args:
        device (torch.device): The device to move the weights to.

    Returns:
        torch.Tensor: A tensor containing the class weights.
    """
    return torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32, device=device)


def weighted_loss(y_pred_logits, y_true):
    """
    Calculates the weighted multi-label logarithmic loss as specified in the competition metric.

    The loss is calculated as:
    L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]

    The losses are summed per sample (patient) and then averaged over the batch.

    Args:
        y_pred_logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
        y_true (torch.Tensor): True labels of shape (batch_size, num_classes).

    Returns:
        torch.Tensor: The calculated scalar loss.
    """
    # Ensure targets are float for BCE calculation
    y_true = y_true.float()

    # Retrieve class weights and move to the correct device
    device = y_pred_logits.device
    weights = get_class_weights(device)

    # Calculate Binary Cross Entropy with Logits
    # reduction='none' computes the loss for every element: L_ij
    # The 'weight' parameter applies the class-specific weight w_j to each column
    element_losses = F.binary_cross_entropy_with_logits(
        y_pred_logits, y_true, weight=weights, reduction="none"
    )

    # Sum the weighted losses across the class dimension (dim=1) to get the loss per patient.
    # Note: Since sum(Config.CLASS_WEIGHTS) == 1.0, this sum effectively represents
    # the weighted average loss for that patient.
    patient_loss = element_losses.sum(dim=1)

    # Average the patient losses across the batch (dim=0)
    return patient_loss.mean()
