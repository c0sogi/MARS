import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Masked Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the MCRMSE metric specifically on the subset of
    target columns defined in Config.SCORED_TARGETS. It effectively masks out
    auxiliary targets (like deg_pH10 and deg_50C) from the gradient optimization
    process, focusing the model on the competition metric.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Retrieve target definitions from Config
        all_targets = Config.ALL_TARGETS
        scored_targets = Config.SCORED_TARGETS

        # Determine the indices of the scored targets within the full target vector
        # Example: If ALL = [A, B, C, D, E] and SCORED = [A, B, D]
        # Indices = [0, 1, 3]
        self.target_indices = [
            i for i, target in enumerate(all_targets) if target in scored_targets
        ]

        # Register indices as a buffer.
        # This ensures the tensor is saved with the model state_dict and
        # automatically moved to the correct device (CPU/GPU) along with the module.
        self.register_buffer(
            "indices", torch.tensor(self.target_indices, dtype=torch.long)
        )

    def forward(self, preds, targets):
        """
        Computes the MCRMSE loss on the scored columns.

        Args:
            preds (torch.Tensor): Predicted values.
                                  Shape: (Batch, SeqLen, Num_All_Targets)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, SeqLen, Num_All_Targets)

        Returns:
            torch.Tensor: A scalar tensor representing the MCRMSE loss.
        """
        # Ensure predictions and targets have the same shape
        # Note: The training loop is expected to slice 'preds' to match the
        # sequence length of 'targets' (e.g., 68) before passing them here.
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch in loss: preds {preds.shape} vs targets {targets.shape}"
            )

        # Select only the scored columns
        # dim=-1 corresponds to the target dimension (channels)
        preds_scored = torch.index_select(preds, dim=-1, index=self.indices)
        targets_scored = torch.index_select(targets, dim=-1, index=self.indices)

        # Compute Squared Errors
        squared_errors = (preds_scored - targets_scored) ** 2

        # Compute Mean Squared Error (MSE) per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
        # leaving a vector of size (Num_Scored_Targets,)
        mse_per_column = torch.mean(squared_errors, dim=(0, 1))

        # Compute Root Mean Squared Error (RMSE) per column
        # Adding a small epsilon for numerical stability during backprop if MSE is 0
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

        # Compute the Mean of the RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
