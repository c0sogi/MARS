import torch
import torch.nn as nn
from library.config import Config
from library.utils import get_class_weights


class MultiTaskLoss(nn.Module):
    """
    Computes the hierarchical multi-task loss for plant classification.
    Combines Species, Genus, and Family losses.
    """

    def __init__(self):
        """
        Initialize the loss criteria.
        Loads class weights for species imbalance handling.
        """
        super(MultiTaskLoss, self).__init__()

        # 1. Load Class Weights for Species
        # These handle the long-tail distribution of the 15,501 species.
        species_weights = get_class_weights(load_cached_data=True)

        # 2. Define Criteria
        # Species: Weighted CrossEntropy with Label Smoothing
        # The weights are passed here; calling .to(device) on this module instance
        # will automatically move these weights to the correct device.
        self.species_criterion = nn.CrossEntropyLoss(
            weight=species_weights, label_smoothing=Config.LABEL_SMOOTHING
        )

        # Genus and Family: Standard CrossEntropy
        # These are auxiliary tasks to structure the feature space.
        self.genus_criterion = nn.CrossEntropyLoss()
        self.family_criterion = nn.CrossEntropyLoss()

        # 3. Hyperparameters
        self.lambda_aux = Config.LAMBDA_AUX

    def forward(self, outputs, targets):
        """
        Compute the total loss.

        Args:
            outputs (dict): Dictionary containing logits from the model:
                            {'species': Tensor, 'genus': Tensor, 'family': Tensor}
            targets (tuple): Tuple containing ground truth indices:
                             (species_targets, genus_targets, family_targets)

        Returns:
            loss (Tensor): The weighted sum of losses.
            loss_dict (dict): Dictionary of individual loss values (floats) for logging.
        """
        # Unpack targets
        species_target, genus_target, family_target = targets

        # Unpack outputs
        species_logits = outputs["species"]
        genus_logits = outputs["genus"]
        family_logits = outputs["family"]

        # Compute individual losses
        loss_species = self.species_criterion(species_logits, species_target)
        loss_genus = self.genus_criterion(genus_logits, genus_target)
        loss_family = self.family_criterion(family_logits, family_target)

        # Compute total weighted loss
        # L_total = L_species + lambda * (L_genus + L_family)
        aux_loss = loss_genus + loss_family
        total_loss = loss_species + (self.lambda_aux * aux_loss)

        # Create dictionary for metrics monitoring (detach values for logging)
        loss_dict = {
            "loss_species": loss_species.item(),
            "loss_genus": loss_genus.item(),
            "loss_family": loss_family.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_dict
