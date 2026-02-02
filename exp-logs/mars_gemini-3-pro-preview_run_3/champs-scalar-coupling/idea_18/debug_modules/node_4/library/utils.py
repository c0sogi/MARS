import torch
from library.config import Config


class GroupStandardizer:
    """
    Handles standardization (z-score normalization) of scalar coupling constants
    independently for each coupling type.
    """

    def __init__(self, device=Config.DEVICE):
        """
        Initialize the standardizer with statistics from Config.

        Args:
            device: The torch device to store the statistics tensors on.
        """
        self.device = device
        # Load stats as (Num_Types,) tensors
        self.means, self.stds = Config.get_coupling_stats_tensor(device)

    def transform(self, values: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values: z = (y - mean) / std

        Args:
            values: Tensor of shape (N,) or (N, 1) containing target values.
            types: Tensor of shape (N,) containing coupling type indices.

        Returns:
            Tensor of standardized values.
        """
        # Ensure correct shape for broadcasting
        if values.dim() == 2 and values.shape[1] == 1:
            values = values.squeeze(1)

        # Gather stats corresponding to the type of each sample
        batch_means = self.means[types]
        batch_stds = self.stds[types]

        z_scores = (values - batch_means) / batch_stds
        return z_scores

    def inverse_transform(
        self, values: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse standardizes values: y = z * std + mean

        Args:
            values: Tensor of shape (N,) or (N, 1) containing standardized values.
            types: Tensor of shape (N,) containing coupling type indices.

        Returns:
            Tensor of values in original scale.
        """
        # Ensure correct shape for broadcasting
        if values.dim() == 2 and values.shape[1] == 1:
            values = values.squeeze(1)

        # Gather stats corresponding to the type of each sample
        batch_means = self.means[types]
        batch_stds = self.stds[types]

        original_values = values * batch_stds + batch_means
        return original_values


class LogMAE:
    """
    Calculates the competition metric: Log of the Mean Absolute Error,
    calculated for each scalar coupling type, and then averaged across types.
    """

    @staticmethod
    def compute(
        preds: torch.Tensor, targets: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the LogMAE metric.

        Metric Formula: 1/T * Sum_t( log( 1/n_t * Sum_i( |y_i - y_pred_i| ) ) )
        where T is number of types, n_t is count of samples for type t.

        Args:
            preds: Tensor of predicted values (original scale).
            targets: Tensor of true values (original scale).
            types: Tensor of coupling type indices.

        Returns:
            Scalar Tensor representing the average LogMAE across present types.
        """
        # Ensure inputs are 1D
        if preds.dim() > 1:
            preds = preds.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        if types.dim() > 1:
            types = types.view(-1)

        # Calculate absolute errors
        abs_errors = torch.abs(preds - targets)

        log_maes = []

        # Identify types present in this batch/dataset
        present_types = torch.unique(types)

        for t in present_types:
            # Mask for the current coupling type
            mask = types == t

            # Calculate MAE for this type
            # We assume mask.sum() > 0 because we iterate over unique present types
            mae_t = abs_errors[mask].mean()

            # Take natural log of the MAE
            # Note: If MAE is 0 (perfect prediction), log is -inf.
            # In practice with floats, exact 0 is rare, but mathematically valid.
            # We use standard torch.log.
            log_maes.append(torch.log(mae_t))

        if not log_maes:
            return torch.tensor(0.0, device=preds.device)

        # Average the log MAEs across types
        return torch.stack(log_maes).mean()
