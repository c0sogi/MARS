import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaussianSmearing(nn.Module):
    """
    Expands a scalar feature (distance or angle) into a vector of Radial Basis Functions.
    Useful for encoding geometric information in graph neural networks.
    """

    def __init__(
        self, start=0.0, stop=5.0, num_gaussians=50, centered=False, learnable=False
    ):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        self.centered = centered

        # Compute offset (means of Gaussians) and width
        offset = torch.linspace(start, stop, num_gaussians)
        # Width is set such that the intersection of two Gaussians is at 0.5 value roughly
        # or simply based on spacing.
        # Using a standard heuristic: width = step
        step = (stop - start) / (num_gaussians - 1) if num_gaussians > 1 else 1.0

        # coeff = -1 / (2 * sigma^2). If sigma = step, then coeff = -0.5 / step^2
        self.coeff = -0.5 / (step**2)

        if learnable:
            self.offset = nn.Parameter(offset)
            self.coeff = nn.Parameter(torch.tensor(self.coeff))
        else:
            self.register_buffer("offset", offset)
            # We keep coeff as a scalar or tensor buffer
            self.register_buffer("coeff_buffer", torch.tensor(self.coeff))

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (..., 1) or (...) containing scalar values.
        Returns:
            Tensor of shape (..., num_gaussians)
        """
        # Ensure dist has a last dimension for broadcasting
        if dist.dim() < 1:
            dist = dist.view(1, 1)
        elif dist.dim() == 1:
            dist = dist.view(-1, 1)
        else:
            dist = dist.unsqueeze(-1)

        # dist: (N, 1), offset: (num_gaussians)
        # diff: (N, num_gaussians)
        diff = dist - self.offset

        # RBF: exp(-gamma * (x - mu)^2)
        coeff = (
            self.coeff
            if hasattr(self, "coeff") and isinstance(self.coeff, nn.Parameter)
            else self.coeff_buffer
        )
        return torch.exp(coeff * torch.pow(diff, 2))


class Standardizer:
    """
    Handles standardization (Z-score normalization) of targets.
    Supports per-group (coupling type) standardization for the primary target
    and global standardization for auxiliary targets.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.means = {}
        self.stds = {}
        self.aux_stats = {}

    def fit(self, df_train, target_col="scalar_coupling_constant", type_col="type"):
        """
        Computes mean and std for each coupling type from the training dataframe.
        """
        # Calculate stats for primary target per type
        groups = df_train.groupby(type_col)[target_col]

        # We use the integer mapping from Config
        type_map = Config.TYPE_MAP

        for t_name, t_idx in type_map.items():
            if t_name in groups.groups:
                group_data = groups.get_group(t_name)
                self.means[t_idx] = group_data.mean()
                self.stds[t_idx] = group_data.std()
            else:
                # Fallback if type not present (unlikely)
                self.means[t_idx] = 0.0
                self.stds[t_idx] = 1.0

        # Convert to tensors for efficient lookup on GPU
        # Create a tensor where index corresponds to type index
        num_types = len(type_map)
        mean_tensor = torch.zeros(num_types, device=self.device)
        std_tensor = torch.ones(num_types, device=self.device)

        for idx in range(num_types):
            mean_tensor[idx] = self.means.get(idx, 0.0)
            std_tensor[idx] = self.stds.get(idx, 1.0)

        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

    def set_aux_stats(self, shielding_mean, shielding_std, charge_mean, charge_std):
        """
        Manually set stats for auxiliary targets.
        """
        self.aux_stats["shielding"] = (shielding_mean, shielding_std)
        self.aux_stats["charge"] = (charge_mean, charge_std)

    def transform(self, values, types):
        """
        Standardizes values based on their type.
        Args:
            values: Tensor (N, ) or (N, 1)
            types: Tensor (N, ) containing integer type indices
        """
        if not torch.is_tensor(values):
            values = torch.tensor(values, device=self.device)
        if not torch.is_tensor(types):
            types = torch.tensor(types, device=self.device)

        values = values.view(-1)
        types = types.view(-1).long()

        means = self.mean_tensor[types]
        stds = self.std_tensor[types]

        return (values - means) / stds

    def inverse_transform(self, values, types):
        """
        Reverts standardization.
        """
        if not torch.is_tensor(values):
            values = torch.tensor(values, device=self.device)
        if not torch.is_tensor(types):
            types = torch.tensor(types, device=self.device)

        values = values.view(-1)
        types = types.view(-1).long()

        means = self.mean_tensor[types]
        stds = self.std_tensor[types]

        return (values * stds) + means

    def transform_aux(self, values, name):
        """
        Standardizes auxiliary targets globally.
        """
        if name not in self.aux_stats:
            return values

        mean, std = self.aux_stats[name]
        if not torch.is_tensor(values):
            values = torch.tensor(values, device=self.device)

        return (values - mean) / std


def calculate_log_mae(preds, targets, types):
    """
    Calculates the Log Mean Absolute Error per type and then averages them.
    This is the competition metric.

    Args:
        preds: Tensor or numpy array of predictions (original scale)
        targets: Tensor or numpy array of ground truth (original scale)
        types: Tensor or numpy array of type indices

    Returns:
        avg_log_mae: Scalar float
        metrics_per_type: Dictionary mapping type name to its log MAE
    """
    # Convert to numpy for metric calculation
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy().flatten()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy().flatten()
    if torch.is_tensor(types):
        types = types.detach().cpu().numpy().flatten()

    preds = preds.flatten()
    targets = targets.flatten()
    types = types.flatten()

    diff = np.abs(preds - targets)

    metrics_per_type = {}
    log_maes = []

    # Inverse map for logging
    inv_type_map = {v: k for k, v in Config.TYPE_MAP.items()}

    unique_types = np.unique(types)

    for t_idx in unique_types:
        mask = types == t_idx
        if np.sum(mask) > 0:
            mae = np.mean(diff[mask])
            # Metric is log(MAE). Using natural log as per standard interpretation of "Log MAE"
            # in this context (often log base e or base 10, usually base e in torch/numpy unless specified).
            # The competition usually specifies log(MAE).
            # To be safe and avoid log(0), we assume MAE > 0.
            log_mae = np.log(mae + 1e-9)

            t_name = inv_type_map.get(t_idx, str(t_idx))
            metrics_per_type[t_name] = log_mae
            log_maes.append(log_mae)

    if len(log_maes) == 0:
        return 0.0, {}

    avg_log_mae = np.mean(log_maes)
    return avg_log_mae, metrics_per_type
