import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification.
    Formula: Loss(x, class) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Args:
            gamma (float): Focusing parameter.
            alpha (Tensor, optional): Weighting factor for each class.
                                      If provided, it should be a 1D Tensor of size C.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Logits of shape (N, C).
            targets (Tensor): Ground truth labels of shape (N).
        """
        # Compute cross entropy loss (log(p_t))
        # We use F.cross_entropy with reduction='none' to get per-sample loss
        # If alpha is provided, it is handled by the weight argument in cross_entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # focal_loss = (1 - p_t)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalLoss(nn.Module):
    """
    Composite loss function for Hierarchical Multi-Task Learning.
    Combines Focal Loss for the Species head and CrossEntropyLoss for Genus and Family heads.
    """

    def __init__(self, weights=None, focal_gamma=2.0, class_weights=None):
        """
        Args:
            weights (dict, optional): Weights for each task head.
                                      Default: {'species': 1.0, 'genus': 0.5, 'family': 0.5}
            focal_gamma (float): Gamma parameter for the species Focal Loss.
            class_weights (Tensor, optional): Pre-computed class weights for the species head
                                              to handle imbalance via Focal Loss alpha.
        """
        super(HierarchicalLoss, self).__init__()

        if weights is None:
            weights = {"species": 1.0, "genus": 0.5, "family": 0.5}
        self.weights = weights

        # Species head uses Focal Loss to handle extreme imbalance (32k classes)
        self.species_loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_weights)

        # Auxiliary heads use standard Cross Entropy
        self.genus_loss_fn = nn.CrossEntropyLoss()
        self.family_loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds (dict): Dictionary containing output logits.
                          Keys: 'species', 'genus', 'family'.
            targets (dict): Dictionary containing target labels.
                            Keys: 'species', 'genus', 'family'.

        Returns:
            total_loss (Tensor): The weighted sum of all losses.
            metrics (dict): Dictionary containing individual loss components for logging.
        """
        # Calculate individual losses
        loss_species = self.species_loss_fn(preds["species"], targets["species"])
        loss_genus = self.genus_loss_fn(preds["genus"], targets["genus"])
        loss_family = self.family_loss_fn(preds["family"], targets["family"])

        # Weighted sum
        w_s = self.weights.get("species", 1.0)
        w_g = self.weights.get("genus", 0.5)
        w_f = self.weights.get("family", 0.5)

        total_loss = (w_s * loss_species) + (w_g * loss_genus) + (w_f * loss_family)

        metrics = {
            "loss_species": loss_species.item(),
            "loss_genus": loss_genus.item(),
            "loss_family": loss_family.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
