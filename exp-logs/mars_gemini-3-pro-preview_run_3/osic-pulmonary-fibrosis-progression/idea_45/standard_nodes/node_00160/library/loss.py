import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """
    Laplace Negative Log Likelihood Loss.

    Designed for the lung function decline prediction task.
    The objective is to minimize the negative log likelihood:
        L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    This corresponds to maximizing the modified metric provided in the competition,
    but without the specific clipping thresholds (70ml sigma clip, 1000ml error clip)
    to ensure gradient flow for outliers and effective learning of uncertainty.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided by the number of
                             elements in the output, 'sum': the output will be summed. Default: 'mean'.
        """
        super(LaplaceNLLLoss, self).__init__()
        self.reduction = reduction

        # Precompute constants for efficiency
        # sqrt(2)
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))
        # ln(sqrt(2)) = 0.5 * ln(2)
        self.register_buffer("log_sqrt_2", torch.log(torch.sqrt(torch.tensor(2.0))))

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the Laplace NLL Loss.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC (mu), shape (Batch,).
            pred_sigma (torch.Tensor): Predicted Confidence (sigma), shape (Batch,).
                                       Must be positive (ensured by model architecture).
            true_fvc (torch.Tensor): Ground Truth FVC, shape (Batch,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure inputs are flattened to 1D vectors for element-wise operations
        pred_fvc = pred_fvc.view(-1)
        pred_sigma = pred_sigma.view(-1)
        true_fvc = true_fvc.view(-1)

        # Calculate absolute error: |y_true - y_pred|
        # We do NOT clip the error to 1000 here, allowing the model to see full magnitude of large errors.
        delta = torch.abs(true_fvc - pred_fvc)

        # Calculate the first term: (sqrt(2) * delta) / sigma
        # pred_sigma is guaranteed positive by the model (softplus + epsilon)
        term1 = (self.sqrt_2 * delta) / pred_sigma

        # Calculate the second term: ln(sqrt(2) * sigma)
        # Expansion: ln(sqrt(2)) + ln(sigma)
        term2 = self.log_sqrt_2 + torch.log(pred_sigma)

        # Total Loss
        loss = term1 + term2

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
