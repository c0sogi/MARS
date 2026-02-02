import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) strictly on:
    1. The scored columns defined in Config.SCORED_COLS.
    2. The valid scored positions defined by Config.PRED_LEN.
    """

    def __init__(self):
        super().__init__()
        self.pred_len = Config.PRED_LEN

        # dynamic determination of indices for scored columns
        # Config.ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Expected indices: [0, 1, 3]
        self.scored_indices = [
            i
            for i, col in enumerate(Config.ALL_TARGET_COLS)
            if col in Config.SCORED_COLS
        ]

        # Register indices as a buffer to ensure they move to device with the module
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (Batch, Length, 5) - Predicted values.
            target: (Batch, Length, 5) - Ground truth values.
            mask: (Batch, Length) - Optional binary mask (1 for valid, 0 for padding).

        Returns:
            loss: Scalar float tensor representing MCRMSE.
        """
        # 1. Slice to valid scored length (0 to 67)
        # We assume the sequence dimension is dim=1
        pred_sliced = pred[:, : self.pred_len, :]
        target_sliced = target[:, : self.pred_len, :]

        # 2. Select only the scored columns
        pred_scored = torch.index_select(pred_sliced, 2, self.scored_indices_tensor)
        target_scored = torch.index_select(target_sliced, 2, self.scored_indices_tensor)

        # 3. Compute Squared Error
        mse = (pred_scored - target_scored) ** 2

        # 4. Apply Masking
        if mask is not None:
            # Slice mask to match scored length
            mask_sliced = mask[:, : self.pred_len]

            # Expand mask for broadcasting: (B, L) -> (B, L, 1)
            mask_sliced = mask_sliced.unsqueeze(2)

            # Apply mask to MSE
            mse = mse * mask_sliced

            # Calculate mean per column, considering only valid positions
            # Sum over Batch (0) and Length (1)
            n_valid = mask_sliced.sum(dim=(0, 1))

            # Avoid division by zero
            mse_col_mean = mse.sum(dim=(0, 1)) / (n_valid + 1e-8)
        else:
            # If no mask, assume all positions in the sliced range are valid
            mse_col_mean = mse.mean(dim=(0, 1))

        # 5. Compute RMSE per column
        rmse_col = torch.sqrt(mse_col_mean)

        # 6. Return Mean of RMSEs (MCRMSE)
        return rmse_col.mean()
