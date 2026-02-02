import torch
import torch.nn as nn
from library.config import Config


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss function for Scalar Coupling Prediction.
    Combines the primary L1 loss for coupling constants with physics-informed
    auxiliary losses (Magnetic Shielding and Mulliken Charges) to regularize
    the learned representations.
    """

    def __init__(self, config=None):
        """
        Args:
            config: Configuration object containing loss weights and flags.
                    Defaults to library.config.Config if None.
        """
        super().__init__()
        self.config = config if config is not None else Config()

        # Use L1 Loss (MAE) for all regression targets as it is robust to outliers
        # and aligns with the competition metric (Log MAE).
        self.l1_loss = nn.L1Loss()

    def forward(self, preds, targets):
        """
        Calculates the weighted sum of primary and auxiliary losses.

        Args:
            preds (dict): Dictionary containing model outputs.
                - 'coupling': Tensor of shape (N, 1)
                - 'shielding': Tensor of shape (N, 9) (Optional)
                - 'charge': Tensor of shape (N, 1) (Optional)
            targets (dict): Dictionary containing ground truth values.
                - 'coupling': Tensor of shape (N,)
                - 'shielding': Tensor of shape (N, 9) (Optional)
                - 'charge': Tensor of shape (N, 1) (Optional)

        Returns:
            tuple: (total_loss, metrics_dict)
                - total_loss (Tensor): Scalar tensor for backpropagation.
                - metrics_dict (dict): Dictionary containing individual loss components (float).
        """
        metrics = {}

        # --- 1. Primary Task: Scalar Coupling ---
        # Ensure shapes match: Preds (N, 1) -> (N,), Targets (N,)
        if "coupling" not in preds or "coupling" not in targets:
            raise ValueError("Predictions and targets must contain 'coupling' key.")

        pred_coupling = preds["coupling"].squeeze()
        target_coupling = targets["coupling"]

        loss_coupling = self.l1_loss(pred_coupling, target_coupling)

        total_loss = loss_coupling
        metrics["loss_coupling"] = loss_coupling.item()

        # --- 2. Auxiliary Tasks ---
        if self.config.USE_AUXILIARY_HEADS:
            aux_weight = self.config.AUX_LOSS_WEIGHT

            # Magnetic Shielding
            # Check if head is active and targets are available
            if (
                preds.get("shielding") is not None
                and targets.get("shielding") is not None
            ):
                loss_shielding = self.l1_loss(preds["shielding"], targets["shielding"])
                total_loss = total_loss + (aux_weight * loss_shielding)
                metrics["loss_shielding"] = loss_shielding.item()

            # Mulliken Charges
            # Check if head is active and targets are available
            if preds.get("charge") is not None and targets.get("charge") is not None:
                loss_charge = self.l1_loss(preds["charge"], targets["charge"])
                total_loss = total_loss + (aux_weight * loss_charge)
                metrics["loss_charge"] = loss_charge.item()

        return total_loss, metrics
