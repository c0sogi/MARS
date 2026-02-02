import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PhysicsAwareLoss(nn.Module):
    """
    Custom objective function for the Dual Graph Network.
    Computes a weighted sum of:
    1. Primary Loss: MAE (L1) of scalar coupling predictions (standardized).
    2. Auxiliary Loss 1: MSE of magnetic shielding tensor predictions.
    3. Auxiliary Loss 2: MSE of Mulliken charge predictions.
    """

    def __init__(self):
        super(PhysicsAwareLoss, self).__init__()
        self.lambda_shielding = Config.LAMBDA_SHIELDING
        self.lambda_charge = Config.LAMBDA_CHARGE

    def forward(self, preds, data):
        """
        Computes the composite loss.

        Args:
            preds (tuple): Tuple containing (pred_coupling, pred_shielding, pred_charges).
                           - pred_coupling: (N_targets, 1)
                           - pred_shielding: (N_nodes, 9)
                           - pred_charges: (N_nodes, 1)
            data (DualGraphData): Batch object containing targets.
                                  - data.y: (N_targets,) or (N_targets, 1) - Standardized coupling constants
                                  - data.aux_shielding: (N_nodes, 9)
                                  - data.aux_charges: (N_nodes, 1)

        Returns:
            torch.Tensor: The scalar total loss.
            dict: Dictionary containing individual loss components for logging.
        """
        pred_coupling, pred_shielding, pred_charges = preds

        # ----------------------------------------------------------------------
        # 1. Primary Loss: Scalar Coupling Constant (MAE)
        # ----------------------------------------------------------------------
        # Ensure target has the same shape as prediction (N, 1)
        target_coupling = data.y
        if target_coupling.dim() == 1:
            target_coupling = target_coupling.view(-1, 1)

        # We use L1 Loss (MAE) on the standardized targets as per strategy.
        # This is robust to outliers and aligns with the competition metric (Log MAE).
        loss_coupling = F.l1_loss(pred_coupling, target_coupling)

        # ----------------------------------------------------------------------
        # 2. Auxiliary Loss: Magnetic Shielding Tensors (MSE)
        # ----------------------------------------------------------------------
        # pred_shielding: (N, 9), data.aux_shielding: (N, 9)
        # MSE is standard for regression of physical vectors/tensors.
        loss_shielding = F.mse_loss(pred_shielding, data.aux_shielding)

        # ----------------------------------------------------------------------
        # 3. Auxiliary Loss: Mulliken Charges (MSE)
        # ----------------------------------------------------------------------
        # pred_charges: (N, 1), data.aux_charges: (N, 1)
        loss_charge = F.mse_loss(pred_charges, data.aux_charges)

        # ----------------------------------------------------------------------
        # Total Loss
        # ----------------------------------------------------------------------
        # Combine losses with configured weights
        total_loss = (
            loss_coupling
            + self.lambda_shielding * loss_shielding
            + self.lambda_charge * loss_charge
        )

        # Metrics for logging
        loss_metrics = {
            "loss_total": total_loss.item(),
            "loss_coupling": loss_coupling.item(),
            "loss_shielding": loss_shielding.item(),
            "loss_charge": loss_charge.item(),
        }

        return total_loss, loss_metrics
