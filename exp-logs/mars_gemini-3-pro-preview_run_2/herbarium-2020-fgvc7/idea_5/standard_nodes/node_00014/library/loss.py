import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import LOSS_WEIGHTS, FOCAL_GAMMA, FOCAL_ALPHA


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    Formula: Loss(x, class) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (Batch, Num_Classes)
            targets (torch.Tensor): (Batch)
        """
        # Compute cross_entropy (which is -log(pt))
        # reduction='none' allows us to apply the focal modulation per sample
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Calculate p_t (probability of the ground truth class)
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalLoss(nn.Module):
    """
    Weighted Multi-Task Loss for Hierarchical Classification.
    Combines Focal Loss for Species and CrossEntropyLoss for Genus/Family.
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()

        # Loss functions
        self.species_loss_fn = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
        self.genus_loss_fn = nn.CrossEntropyLoss()
        self.family_loss_fn = nn.CrossEntropyLoss()

        # Weights from config
        self.weights = LOSS_WEIGHTS

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary of logits {'species': ..., 'genus': ..., 'family': ...}
            targets (tuple): Tuple of target tensors (species_target, genus_target, family_target)

        Returns:
            total_loss (torch.Tensor): Weighted sum of losses
            metrics (dict): Dictionary of scalar loss values for logging
        """
        # Unpack outputs and targets
        species_logits = outputs["species"]
        genus_logits = outputs["genus"]
        family_logits = outputs["family"]

        species_targets, genus_targets, family_targets = targets

        # Compute losses
        loss_species = self.species_loss_fn(species_logits, species_targets)
        loss_genus = self.genus_loss_fn(genus_logits, genus_targets)
        loss_family = self.family_loss_fn(family_logits, family_targets)

        # Weighted sum
        total_loss = (
            self.weights["species"] * loss_species
            + self.weights["genus"] * loss_genus
            + self.weights["family"] * loss_family
        )

        # Return total loss and individual metrics
        metrics = {
            "loss_species": loss_species.item(),
            "loss_genus": loss_genus.item(),
            "loss_family": loss_family.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
