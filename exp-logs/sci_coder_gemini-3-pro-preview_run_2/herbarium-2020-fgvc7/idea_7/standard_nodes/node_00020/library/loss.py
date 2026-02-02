import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)

    This implementation works directly with logits for numerical stability.
    It dynamically down-weights well-classified examples to focus on hard, rare classes.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            gamma (float): Focusing parameter. Higher values down-weight easy examples more.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Predicted logits of shape (Batch, Num_Classes).
            targets (Tensor): Ground truth labels of shape (Batch).

        Returns:
            Tensor: Calculated loss.
        """
        # Compute standard cross entropy: -log(p_t)
        # using reduction='none' to preserve per-sample loss for the focal term application
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Calculate probabilities p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Calculate Focal term: (1 - p_t)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)

        # Final Focal Loss
        loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HierarchicalLoss(nn.Module):
    """
    Composite loss function for Hierarchical Plant Species Classification.

    Combines:
    1. Focal Loss for the Species head (to handle long-tail distribution).
    2. Cross Entropy Loss for Genus and Family heads (auxiliary supervision).
    """

    def __init__(self, config: Config, weights: dict = None):
        """
        Args:
            config (Config): Configuration object containing hyperparameters.
            weights (dict, optional): Weights for each task.
                                      Defaults to {'species': 1.0, 'genus': 0.5, 'family': 0.5}.
        """
        super(HierarchicalLoss, self).__init__()
        self.config = config

        # Default weights prioritize the main species task while using hierarchy for regularization
        if weights is None:
            self.weights = {"species": 1.0, "genus": 0.5, "family": 0.5}
        else:
            self.weights = weights

        # Species head uses Focal Loss
        self.species_loss_fn = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA)

        # Auxiliary heads use standard Cross Entropy
        self.aux_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs: tuple, targets: tuple):
        """
        Computes the weighted sum of losses from all heads.

        Args:
            outputs (tuple): (species_logits, genus_logits, family_logits) from the model.
            targets (tuple): (species_targets, genus_targets, family_targets) from the dataset.

        Returns:
            total_loss (Tensor): The combined loss for backpropagation.
            metrics (dict): Dictionary containing individual loss components for logging.
        """
        species_logits, genus_logits, family_logits = outputs
        species_targets, genus_targets, family_targets = targets

        # 1. Calculate Species Loss (Focal)
        loss_species = self.species_loss_fn(species_logits, species_targets)

        # 2. Calculate Genus Loss (Cross Entropy)
        loss_genus = self.aux_loss_fn(genus_logits, genus_targets)

        # 3. Calculate Family Loss (Cross Entropy)
        loss_family = self.aux_loss_fn(family_logits, family_targets)

        # 4. Weighted Combination
        total_loss = (
            self.weights["species"] * loss_species
            + self.weights["genus"] * loss_genus
            + self.weights["family"] * loss_family
        )

        # Return metrics for logging purposes
        metrics = {
            "loss_species": loss_species.item(),
            "loss_genus": loss_genus.item(),
            "loss_family": loss_family.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
