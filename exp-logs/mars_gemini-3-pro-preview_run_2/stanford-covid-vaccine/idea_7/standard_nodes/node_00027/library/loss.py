import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Masked Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the MCRMSE only for the columns specified
    in Config.SCORED_COLS, effectively masking out auxiliary targets from
    the optimization process. It also handles sequence length mismatch by
    slicing predictions to match the target length (Config.SEQ_SCORED).
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS

        # Determine indices of columns to score
        # Example: If targets are [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        # and scored are [reactivity, deg_Mg_pH10, deg_Mg_50C]
        # indices will be [0, 1, 3].
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        # Register indices as a buffer so they move to the correct device automatically
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions.
                                  Shape: (Batch, Seq_Len_Pred, Num_Targets)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Len_Target, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Align Sequence Lengths
        # The model might predict 107 positions, but targets usually cover the first 68.
        # We slice predictions to match the target sequence length.
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]
        elif preds.shape[1] < targets.shape[1]:
            # This case shouldn't typically happen given the task setup,
            # but we slice targets to match preds just in case.
            targets = targets[:, : preds.shape[1], :]

        # 2. Select Scored Columns
        # Filter both tensors to keep only the columns relevant for the metric.
        preds_scored = torch.index_select(
            preds, dim=2, index=self.scored_indices_tensor
        )
        targets_scored = torch.index_select(
            targets, dim=2, index=self.scored_indices_tensor
        )

        # 3. Compute MSE per Column
        # We compute squared errors.
        squared_errors = (preds_scored - targets_scored) ** 2

        # Handle NaNs in targets if any (though data is expected to be dense)
        # We create a mask of valid elements.
        mask = ~torch.isnan(targets_scored)

        # Apply mask: zero out errors where target is NaN
        squared_errors = squared_errors * mask.float()

        # Calculate Mean Squared Error for each column independently.
        # Sum of errors per column / Count of valid elements per column
        # Sum over Batch (dim 0) and Sequence (dim 1)
        sum_squared_errors = torch.sum(squared_errors, dim=(0, 1))
        counts = torch.sum(mask.float(), dim=(0, 1))

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)

        mse_per_column = sum_squared_errors / counts

        # 4. Compute RMSE per Column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Average RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
