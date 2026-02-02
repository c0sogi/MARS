import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the target class.
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float): Weighting factor for the rare class (0 < alpha < 1).
                           If -1, alpha weighting is disabled.
            gamma (float): Focusing parameter (gamma >= 0).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits of shape (N, C)
            targets (torch.Tensor): Ground truth labels of shape (N,)
        """
        # standard cross entropy loss (log(p_t))
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting if configured
        if self.alpha >= 0:
            # We want alpha for the target class.
            # Since this is multi-class, alpha is usually applied as a scalar or vector.
            # In the standard definition for binary: -alpha * (1-p)^gamma * log(p) for class 1
            # For multi-class, it's often simplified to just a scalar factor or class weights.
            # Here we apply the scalar alpha uniformly as a scaling factor for the loss magnitude
            # relative to other losses in the system, or we can implement it strictly.
            # Given the Config.FOCAL_ALPHA is a single float (0.25), we apply it as a scalar multiplier.
            loss = self.alpha * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HierarchicalLoss(nn.Module):
    """
    Composite loss function for the hierarchical model.
    Combines Focal Loss for Species and CrossEntropyLoss for Genus and Family.
    """

    def __init__(
        self,
        weight_species=Config.LOSS_WEIGHT_SPECIES,
        weight_genus=Config.LOSS_WEIGHT_GENUS,
        weight_family=Config.LOSS_WEIGHT_FAMILY,
    ):
        super(HierarchicalLoss, self).__init__()

        self.weight_species = weight_species
        self.weight_genus = weight_genus
        self.weight_family = weight_family

        # Species head uses Focal Loss due to extreme imbalance and long tail
        self.species_loss_fn = FocalLoss(
            alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA
        )

        # Genus and Family heads use standard Cross Entropy
        # These tasks are easier and less imbalanced
        self.genus_loss_fn = nn.CrossEntropyLoss()
        self.family_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                            - 'species': logits for species
                            - 'genus': logits for genus
                            - 'family': logits for family
            targets (dict): Dictionary containing ground truth labels:
                            - 'species_id': tensor of species labels
                            - 'genus_id': tensor of genus labels
                            - 'family_id': tensor of family labels

        Returns:
            total_loss (torch.Tensor): The weighted sum of losses.
            metrics (dict): Dictionary of individual loss components (detached) for logging.
        """

        # 1. Species Loss
        species_logits = outputs["species"]
        species_targets = targets["species_id"]
        loss_species = self.species_loss_fn(species_logits, species_targets)

        # 2. Genus Loss
        genus_logits = outputs["genus"]
        genus_targets = targets["genus_id"]
        loss_genus = self.genus_loss_fn(genus_logits, genus_targets)

        # 3. Family Loss
        family_logits = outputs["family"]
        family_targets = targets["family_id"]
        loss_family = self.family_loss_fn(family_logits, family_targets)

        # 4. Weighted Sum
        total_loss = (
            self.weight_species * loss_species
            + self.weight_genus * loss_genus
            + self.weight_family * loss_family
        )

        # Dictionary for logging
        metrics = {
            "loss_total": total_loss.detach().item(),
            "loss_species": loss_species.detach().item(),
            "loss_genus": loss_genus.detach().item(),
            "loss_family": loss_family.detach().item(),
        }

        return total_loss, metrics
