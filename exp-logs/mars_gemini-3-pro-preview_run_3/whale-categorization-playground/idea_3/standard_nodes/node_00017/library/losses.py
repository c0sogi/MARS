import torch
import torch.nn as nn
import math
from library.config import CFG


class CurricularFaceLoss(nn.Module):
    """
    Implements CurricularFace Loss (CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition).

    This loss dynamically adjusts the margin based on the difficulty of samples (curriculum learning).
    It emphasizes hard samples by modulating the cosine similarities of negative classes that are
    closer to the query than the margin-enforced ground truth.

    Reference: https://arxiv.org/abs/2004.00288
    """

    def __init__(self, s=None, m=None):
        """
        Args:
            s (float, optional): Scale factor (inverse temperature). Defaults to CFG.s.
            m (float, optional): Margin value. Defaults to CFG.m.
        """
        super(CurricularFaceLoss, self).__init__()
        self.s = s if s is not None else CFG.s
        self.m = m if m is not None else CFG.m

        # Register t as a buffer so it's part of the state_dict but not a learnable parameter.
        # t represents the moving average of the mean margin-based target cosine similarity.
        # It serves as the curriculum difficulty threshold.
        self.register_buffer("t", torch.zeros(1))

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Forward pass for CurricularFace Loss.

        Args:
            logits (torch.Tensor): Cosine logits (cosine similarities) with shape (batch_size, num_classes).
                                   Values should be in range [-1, 1].
            labels (torch.Tensor): Ground truth labels with shape (batch_size).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # 1. Get the cosine similarity of the ground truth class: cos(theta_yi)
        # shape: (batch_size, 1)
        target_logits = logits.gather(1, labels.view(-1, 1))

        # 2. Compute the margin-enforced target: cos(theta_yi + m)
        # We use the trigonometric identity: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)

        # Clamp target logits for numerical stability in acos/sqrt
        target_logits_clamped = target_logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logits_clamped, 2))
        cos_theta_m = target_logits_clamped * math.cos(self.m) - sin_theta * math.sin(
            self.m
        )

        # 3. Update the curriculum parameter t (moving average)
        # Only update during training
        if self.training:
            with torch.no_grad():
                self.t = 0.99 * self.t + 0.01 * cos_theta_m.mean()

        # 4. Mask hard negative samples
        # Hard samples are negative classes where cos(theta_j) > cos(theta_yi + m)
        # i.e., the model thinks this negative class is more similar than the margin-enforced ground truth.

        # Create a mask for all logits
        mask = torch.ones_like(logits, dtype=torch.bool)
        # Exclude the ground truth index from the mask (we only modulate negatives)
        mask.scatter_(1, labels.view(-1, 1), False)

        # Determine which negatives are "hard"
        # We compare every logit against the specific target_logit_m for that sample (broadcasting works here)
        hard_mask = mask & (logits > cos_theta_m)

        # 5. Apply Modulation to hard negatives
        # Formula: N(t, theta_j) = theta_j * (t + theta_j)
        # For easy negatives (or ground truth), we keep them as is initially.
        # Note: self.t is a scalar tensor buffer

        logits_modulated = torch.where(hard_mask, logits * (self.t + logits), logits)

        # 6. Replace the ground truth logit with the margin-enforced one
        # We put cos(theta_yi + m) back into the correct positions
        final_logits = logits_modulated.scatter(1, labels.view(-1, 1), cos_theta_m)

        # 7. Scale by s and compute Cross Entropy
        final_logits = final_logits * self.s

        loss = self.ce_loss(final_logits, labels)

        return loss
