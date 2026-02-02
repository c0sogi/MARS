import torch
import torch.nn as nn
from library.config import Config


class LaplaceNLLLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, we minimize the negative metric (Loss):
        Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()
        self.error_clip = Config.ERROR_CLIP
        self.confidence_clip = Config.CONFIDENCE_CLIP
        # Register sqrt(2) as a buffer to avoid recomputing and ensure device consistency
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, target):
        """
        Calculate the Laplace NLL Loss.

        Args:
            preds (torch.Tensor): Tensor of shape (batch_size, 2).
                                  - Column 0: Predicted FVC (ml)
                                  - Column 1: Predicted Confidence/Sigma (ml)
            target (torch.Tensor): Tensor of shape (batch_size,) or (batch_size, 1).
                                   - True FVC (ml)

        Returns:
            torch.Tensor: Scalar tensor containing the mean loss over the batch.
        """
        # Separate predictions
        fvc_pred = preds[:, 0]
        sigma_pred = preds[:, 1]

        # Flatten target to match fvc_pred shape (batch_size,)
        fvc_true = target.view(-1)

        # 1. Apply Confidence Clipping: sigma_clipped = max(sigma, 70)
        sigma_clipped = torch.clamp(sigma_pred, min=self.confidence_clip)

        # 2. Apply Error Thresholding: delta = min(|True - Pred|, 1000)
        abs_error = torch.abs(fvc_true - fvc_pred)
        delta = torch.clamp(abs_error, max=self.error_clip)

        # 3. Calculate Loss Terms
        # Term 1: (sqrt(2) * delta) / sigma_clipped
        term_1 = (self.sqrt_2 * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * sigma_clipped)
        term_2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Total Loss = Term 1 + Term 2
        loss = term_1 + term_2

        # Return mean loss over the batch
        return torch.mean(loss)
