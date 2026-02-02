import torch
import torch.nn as nn
import numpy as np


class MCRMSELoss(nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # The competition scores 3 specific columns out of the 5 provided in training.
        # Indices based on metadata:
        # 0: reactivity (Scored)
        # 1: deg_Mg_pH10 (Scored)
        # 2: deg_pH10 (Not Scored)
        # 3: deg_Mg_50C (Scored)
        # 4: deg_50C (Not Scored)
        self.scored_indices = [0, 1, 3]
        self.seq_scored = 68
        self.eps = 1e-6

    def forward(self, inputs, targets):
        """
        Calculate MCRMSE loss.

        Args:
            inputs: Predictions tensor of shape (batch_size, seq_len, 5)
            targets: Ground truth tensor of shape (batch_size, seq_len, 5)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Slice to the scored sequence length (first 68 positions)
        # and select only the scored columns.
        pred_scored = inputs[:, : self.seq_scored, self.scored_indices]
        true_scored = targets[:, : self.seq_scored, self.scored_indices]

        # Calculate MSE per column (averaging over batch and sequence positions)
        # dim=(0, 1) averages over batch and sequence dimensions, leaving (3,)
        mse = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 1))

        # Calculate RMSE per column with epsilon for stability
        rmse = torch.sqrt(mse + self.eps)

        # Average RMSE across the 3 scored columns to get MCRMSE
        loss = torch.mean(rmse)

        return loss


class GlobalMetricsTracker:
    def __init__(self):
        # Configuration matches the Loss class
        self.scored_indices = [0, 1, 3]
        self.col_names = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.seq_scored = 68
        self.reset()

    def reset(self):
        """Reset internal accumulators."""
        self.sse = {name: 0.0 for name in self.col_names}
        self.counts = {name: 0 for name in self.col_names}

    def update(self, preds, targets):
        """
        Update metrics with a new batch of predictions and targets.

        Args:
            preds: Predictions (batch, seq_len, 5), Tensor or numpy array
            targets: Ground truth (batch, seq_len, 5), Tensor or numpy array
        """
        # Convert to numpy if tensors
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Slice to scored length
        p = preds[:, : self.seq_scored, :]
        t = targets[:, : self.seq_scored, :]

        # Accumulate SSE for each scored column
        for i, idx in enumerate(self.scored_indices):
            col_name = self.col_names[i]

            p_col = p[:, :, idx]
            t_col = t[:, :, idx]

            # Calculate squared errors
            squared_diff = (p_col - t_col) ** 2

            self.sse[col_name] += np.sum(squared_diff)
            self.counts[col_name] += squared_diff.size

    def compute(self):
        """
        Compute the global metrics based on accumulated data.

        Returns:
            dict: Dictionary containing RMSE per column and the global MCRMSE.
        """
        metrics = {}
        total_rmse = 0.0

        for name in self.col_names:
            count = self.counts[name]
            if count > 0:
                mse = self.sse[name] / count
                rmse = np.sqrt(mse)
            else:
                rmse = 0.0

            metrics[f"rmse_{name}"] = rmse
            total_rmse += rmse

        # MCRMSE is the mean of the column-wise RMSEs
        metrics["mcrmse"] = total_rmse / len(self.col_names)
        return metrics
