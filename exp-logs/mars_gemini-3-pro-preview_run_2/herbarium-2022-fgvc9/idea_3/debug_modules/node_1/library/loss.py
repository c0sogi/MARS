import torch
import torch.nn as nn
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    Computes the Cross Entropy Loss for the ArcFace head.

    Note: The additive angular margin penalty is applied within the
    ArcFaceLayer in model.py (forward pass) when labels are provided.
    This module computes the standard Cross Entropy on those penalized logits.
    """

    def __init__(self):
        super(ArcFaceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): Logits from the model (batch_size, num_classes).
                                   These are expected to be margin-penalized if training.
            labels (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(logits, labels)


class HierarchicalMultiTaskLoss(nn.Module):
    """
    Composite loss function for the Hierarchical Metric Learning Network.

    Aggregates:
    1. Species Loss (ArcFace/CrossEntropy)
    2. Genus Loss (Auxiliary CrossEntropy)
    3. Family Loss (Auxiliary CrossEntropy)

    The aggregation is weighted by parameters defined in Config.
    """

    def __init__(self):
        super(HierarchicalMultiTaskLoss, self).__init__()
        self.species_criterion = ArcFaceLoss()
        self.aux_criterion = nn.CrossEntropyLoss()

        # Load weights from Config
        self.w_species = Config.LOSS_WEIGHT_SPECIES
        self.w_genus = Config.LOSS_WEIGHT_GENUS
        self.w_family = Config.LOSS_WEIGHT_FAMILY

    def forward(self, outputs, targets):
        """
        Computes the weighted multi-task loss.

        Args:
            outputs (dict): Dictionary containing model outputs with keys:
                            - 'species': Tensor [B, num_species]
                            - 'genus': Tensor [B, num_genera]
                            - 'family': Tensor [B, num_families]
            targets (tuple or list): Sequence containing:
                            - species_labels: Tensor [B]
                            - genus_labels: Tensor [B]
                            - family_labels: Tensor [B]

        Returns:
            tuple: (total_loss, metrics_dict)
                   - total_loss (torch.Tensor): The scalar weighted loss for backprop.
                   - metrics_dict (dict): Dictionary of individual loss components (floats).
        """
        # Unpack targets
        species_labels, genus_labels, family_labels = targets

        # Extract logits
        species_logits = outputs["species"]
        genus_logits = outputs["genus"]
        family_logits = outputs["family"]

        # Compute individual losses
        loss_species = self.species_criterion(species_logits, species_labels)
        loss_genus = self.aux_criterion(genus_logits, genus_labels)
        loss_family = self.aux_criterion(family_logits, family_labels)

        # Compute weighted total loss
        total_loss = (
            (self.w_species * loss_species)
            + (self.w_genus * loss_genus)
            + (self.w_family * loss_family)
        )

        # Create metrics dictionary for logging
        metrics = {
            "loss_total": total_loss.item(),
            "loss_species": loss_species.item(),
            "loss_genus": loss_genus.item(),
            "loss_family": loss_family.item(),
        }

        return total_loss, metrics
