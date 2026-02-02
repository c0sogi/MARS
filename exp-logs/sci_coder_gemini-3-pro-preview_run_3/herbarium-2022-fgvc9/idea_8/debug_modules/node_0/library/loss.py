import torch
import torch.nn as nn


class HierarchicalLoss(nn.Module):
    """
    Multi-task objective function for Hierarchical Plant Classification.

    Computes the CrossEntropyLoss with label smoothing for Species, Genus, and Family heads,
    and aggregates them into a single scalar loss using a weighted sum.
    """

    def __init__(self, genus_weight=0.1, family_weight=0.1, label_smoothing=0.1):
        """
        Args:
            genus_weight (float): Weight applied to the genus head loss. Default is 0.1.
            family_weight (float): Weight applied to the family head loss. Default is 0.1.
            label_smoothing (float): Label smoothing factor for CrossEntropyLoss. Default is 0.1.
        """
        super(HierarchicalLoss, self).__init__()
        self.genus_weight = genus_weight
        self.family_weight = family_weight

        # Use CrossEntropyLoss with label smoothing for all heads
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        """
        Computes the total weighted loss.

        Args:
            outputs (dict): Dictionary containing logits from the model.
                            Expected keys: 'species', 'genus', 'family'.
            targets (tuple): Tuple containing ground truth labels in the order:
                             (species_labels, genus_labels, family_labels).

        Returns:
            torch.Tensor: The aggregated scalar total loss.
        """
        # Unpack targets
        species_labels, genus_labels, family_labels = targets

        # Compute loss for the primary task (Species)
        loss_species = self.criterion(outputs["species"], species_labels)

        # Compute loss for auxiliary tasks (Genus, Family)
        loss_genus = self.criterion(outputs["genus"], genus_labels)
        loss_family = self.criterion(outputs["family"], family_labels)

        # Aggregate losses: L_total = L_species + w_g * L_genus + w_f * L_family
        total_loss = (
            loss_species
            + (self.genus_weight * loss_genus)
            + (self.family_weight * loss_family)
        )

        return total_loss
