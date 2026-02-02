import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the Root Mean Squared Error (RMSE) for each
    target column independently and then returns the average of these RMSEs.

    It is designed to handle the specific requirement of the RNA degradation task
    where predictions are generated for the full sequence (e.g., 107 bases) but
    ground truth is provided only for a subset (e.g., first 68 bases).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Computes the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values from the model.
                Expected shape: (Batch_Size, Seq_Len_Pred, Num_Targets).
                Example: (32, 107, 5).
            targets (torch.Tensor): Ground truth values.
                Expected shape: (Batch_Size, Seq_Len_Tgt, Num_Targets).
                Example: (32, 68, 5).

        Returns:
            torch.Tensor: A scalar tensor representing the mean of column-wise RMSEs.
        """
        # 1. Align Sequence Lengths
        # The model predicts the full sequence (107), but targets exist only for
        # the scored positions (68). We slice the inputs to match the targets.
        if inputs.shape[1] > targets.shape[1]:
            inputs = inputs[:, : targets.shape[1], :]

        # 2. Compute Squared Errors
        # Shape: (Batch_Size, Seq_Len_Tgt, Num_Targets)
        squared_diff = (inputs - targets) ** 2

        # 3. Compute Mean Squared Error (MSE) per column
        # We average over the Batch (dim=0) and Sequence (dim=1) dimensions.
        # Result Shape: (Num_Targets,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute Root Mean Squared Error (RMSE) per column
        # Result Shape: (Num_Targets,)
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)  # epsilon for stability

        # 5. Compute Mean of RMSEs (MCRMSE)
        # Result Shape: Scalar
        loss = torch.mean(rmse_per_column)

        return loss
