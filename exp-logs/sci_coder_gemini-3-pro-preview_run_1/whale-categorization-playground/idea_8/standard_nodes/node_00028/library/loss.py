import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    This loss function regularizes the model by replacing the hard one-hot target
    distribution with a smoothed distribution. This prevents the model from becoming
    over-confident and improves generalization, especially in classification tasks
    with many classes or noisy labels.

    Formula:
        loss = (1 - epsilon) * H(q, p) + epsilon * H(u, p)
        where:
            p = predicted probabilities (softmax)
            q = ground truth distribution (one-hot)
            u = uniform distribution (1/K)
            H = Cross Entropy
    """

    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing (float): The label smoothing factor (epsilon).
                               Must be between 0.0 and 1.0. Default is 0.1.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing value must be in [0, 1)"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        """
        Computes the Label Smoothing Cross Entropy loss.

        Args:
            x (torch.Tensor): Logits from the model of shape (Batch, NumClasses).
            target (torch.Tensor): Ground truth class indices of shape (Batch,).

        Returns:
            torch.Tensor: Scalar tensor representing the mean loss over the batch.
        """
        # Compute log probabilities (log_softmax is numerically more stable)
        logprobs = F.log_softmax(x, dim=-1)

        # 1. Compute NLL Loss (Cross Entropy with hard targets)
        # gather extracts the log-prob of the true class
        # target.unsqueeze(1) changes shape from (B) to (B, 1) for gather
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)  # Shape: (B)

        # 2. Compute Smooth Loss (Cross Entropy with uniform distribution)
        # This is equivalent to - sum(1/K * log(p_i)) = - mean(log(p))
        smooth_loss = -logprobs.mean(dim=-1)  # Shape: (B)

        # 3. Combine components
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Return mean loss over the batch
        return loss.mean()
