import torch
import torch.nn as nn
from library.config import Config


class HierarchicalLoss(nn.Module):
    """
    Custom loss module for Hierarchical Plant Classification.
    Computes a weighted sum of CrossEntropyLoss for Species, Genus, and Family heads.
    """

    def __init__(
        self,
        weight_species=Config.LOSS_WEIGHT_SPECIES,
        weight_genus=Config.LOSS_WEIGHT_GENUS,
        weight_family=Config.LOSS_WEIGHT_FAMILY,
        label_smoothing=Config.LABEL_SMOOTHING,
    ):
        """
        Args:
            weight_species (float): Weight for the species loss (primary task).
            weight_genus (float): Weight for the genus loss (auxiliary task).
            weight_family (float): Weight for the family loss (auxiliary task).
            label_smoothing (float): Label smoothing factor (0.0 to 1.0).
        """
        super(HierarchicalLoss, self).__init__()

        self.weight_species = weight_species
        self.weight_genus = weight_genus
        self.weight_family = weight_family

        # Initialize separate loss criteria for each head
        # This allows for future flexibility (e.g., different smoothing per head)
        self.species_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.genus_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.family_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        """
        Computes the total weighted loss.

        Args:
            outputs (dict): Dictionary containing model output logits.
                            Keys: 'species', 'genus', 'family'.
            targets (dict): Dictionary containing target tensors.
                            Keys: 'species', 'genus', 'family'.

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # 1. Species Loss (Primary)
        loss_species = self.species_criterion(outputs["species"], targets["species"])

        # 2. Genus Loss (Auxiliary)
        loss_genus = self.genus_criterion(outputs["genus"], targets["genus"])

        # 3. Family Loss (Auxiliary)
        loss_family = self.family_criterion(outputs["family"], targets["family"])

        # Weighted Sum
        total_loss = (
            (self.weight_species * loss_species)
            + (self.weight_genus * loss_genus)
            + (self.weight_family * loss_family)
        )

        return total_loss
