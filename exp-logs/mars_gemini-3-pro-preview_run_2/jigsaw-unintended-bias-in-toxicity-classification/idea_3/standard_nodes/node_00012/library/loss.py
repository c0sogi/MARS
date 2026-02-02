import torch
import torch.nn as nn
from library.config import Config


class AggressiveMultiTaskLoss(nn.Module):
    """
    Implements the objective function for the multi-task learning strategy.
    Combines Binary Cross Entropy (BCE) for the main toxicity target and
    auxiliary identity attributes.
    """

    def __init__(self):
        super().__init__()
        # Load weights from configuration
        self.toxicity_weight = Config.TOXICITY_WEIGHT
        self.identity_weight = Config.IDENTITY_WEIGHT

        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        toxicity_logits: torch.Tensor,
        identity_logits: torch.Tensor,
        targets: torch.Tensor,
        identities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the weighted multi-task loss.

        Args:
            toxicity_logits (torch.Tensor): Logits from the toxicity head. Shape: (batch_size, 1)
            identity_logits (torch.Tensor): Logits from the identity head. Shape: (batch_size, num_identities)
            targets (torch.Tensor): Ground truth toxicity labels. Shape: (batch_size,) or (batch_size, 1)
            identities (torch.Tensor): Ground truth identity labels. Shape: (batch_size, num_identities)

        Returns:
            torch.Tensor: The scalar weighted total loss.
        """
        # Ensure targets have the same shape as logits (batch_size, 1)
        # DataLoader typically yields targets as (batch_size,), but Linear layer outputs (batch_size, 1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # 1. Main Task Loss: Toxicity
        toxicity_loss = self.loss_fn(toxicity_logits, targets)

        # 2. Auxiliary Task Loss: Identity Attributes
        # Identity logits and labels should already be aligned in shape (batch_size, num_identities)
        identity_loss = self.loss_fn(identity_logits, identities)

        # 3. Combine Losses
        # We use a weighted sum to control the relative importance of the auxiliary task.
        # A higher identity_weight forces the shared backbone to learn features that distinguish
        # specific identities, helping to disentangle them from the concept of toxicity.
        total_loss = (self.toxicity_weight * toxicity_loss) + (
            self.identity_weight * identity_loss
        )

        return total_loss
