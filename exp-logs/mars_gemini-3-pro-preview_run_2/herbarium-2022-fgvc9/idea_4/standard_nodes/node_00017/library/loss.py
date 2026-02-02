import torch
import torch.nn as nn
from library.config import Config


class HierarchicalLoss(nn.Module):
    """
    Implements a multi-task objective function for the Cascaded Taxonomic Network.
    It calculates CrossEntropyLoss for the family and genus auxiliary heads and
    combines them with the loss from the species head (ArcFace).
    """

    def __init__(self, lambda_genus=None, lambda_family=None):
        """
        Args:
            lambda_genus (float, optional): Weight for the genus classification loss.
                                          Defaults to Config.LAMBDA_GENUS.
            lambda_family (float, optional): Weight for the family classification loss.
                                           Defaults to Config.LAMBDA_FAMILY.
        """
        super(HierarchicalLoss, self).__init__()
        self.lambda_genus = (
            lambda_genus if lambda_genus is not None else Config.LAMBDA_GENUS
        )
        self.lambda_family = (
            lambda_family if lambda_family is not None else Config.LAMBDA_FAMILY
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Computes the weighted sum of losses.

        Args:
            outputs (tuple): A tuple containing (species_logits, genus_logits, family_logits).
                             species_logits are expected to be the output of the ArcFace layer
                             (already scaled and with margins applied during training).
            targets (tuple): A tuple containing (species_labels, genus_labels, family_labels).

        Returns:
            torch.Tensor: The total weighted loss.
        """
        # Unpack inputs
        sp_logits, gn_logits, fm_logits = outputs
        sp_labels, gn_labels, fm_labels = targets

        # Compute individual losses
        # Note: ArcFace logits are compatible with CrossEntropyLoss as it applies Softmax internally
        loss_sp = self.criterion(sp_logits, sp_labels)
        loss_gn = self.criterion(gn_logits, gn_labels)
        loss_fm = self.criterion(fm_logits, fm_labels)

        # Weighted sum
        total_loss = (
            loss_sp + (self.lambda_genus * loss_gn) + (self.lambda_family * loss_fm)
        )

        return total_loss
