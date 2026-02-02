import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from library.config import Config
from library.utils import AverageMeter


def get_normalization_tensors(standardizer, device):
    """
    Converts standardizer statistics to tensors for vectorized normalization on GPU.
    """
    types_list = Config.COUPLING_TYPES
    means = []
    stds = []

    for t in types_list:
        m, s = standardizer.get_params(t)
        means.append(m)
        stds.append(s)

    means_tensor = torch.tensor(means, device=device, dtype=torch.float)
    stds_tensor = torch.tensor(stds, device=device, dtype=torch.float)

    return means_tensor, stds_tensor


def train_one_epoch(model, loader, optimizer, device, standardizer, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    # Pre-load normalization stats to device
    means_tensor, stds_tensor = get_normalization_tensors(standardizer, device)

    for batch in loader:
        batch = batch.to(device)

        # 1. Prepare Targets (Normalize)
        # batch.y is the raw scalar coupling constant
        # batch.coupling_type is the index of the type

        # Flatten tensors to ensure matching shapes
        targets_raw = batch.y.view(-1)
        type_indices = batch.coupling_type.view(-1)

        # Vectorized normalization: (y - mean[type]) / std[type]
        batch_means = means_tensor[type_indices]
        batch_stds = stds_tensor[type_indices]
        targets_norm = (targets_raw - batch_means) / batch_stds

        # 2. Forward Pass
        optimizer.zero_grad()
        preds = model(batch).view(-1)

        # 3. Loss Calculation (MAE on normalized data)
        loss = torch.abs(preds - targets_norm).mean()

        # 4. Backward Pass
        loss.backward()

        # 5. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # 6. Optimization Step
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), batch.num_graphs)

    return loss_meter.avg


def evaluate(model, loader, device, standardizer):
    """
    Evaluates the model on the validation set.
    Metric: Log of the Mean Absolute Error, averaged across types.
    """
    model.eval()

    # Accumulators for calculation of MAE per type
    # We sum the absolute errors and the counts, then compute mean at the end
    type_abs_error_sum = defaultdict(float)
    type_counts = defaultdict(int)

    # List of type names for mapping indices back to strings
    type_names = Config.COUPLING_TYPES

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            preds_norm = model(batch).view(-1)

            # Inverse transform predictions to original scale
            # We use the standardizer's vectorized logic or implement it here manually
            # Using manual implementation for speed/batching within GPU if possible,
            # but standardizer.inverse_transform works on numpy.
            # Let's stay on GPU for aggregation.

            means_tensor, stds_tensor = get_normalization_tensors(standardizer, device)
            type_indices = batch.coupling_type.view(-1)

            batch_means = means_tensor[type_indices]
            batch_stds = stds_tensor[type_indices]

            preds_raw = preds_norm * batch_stds + batch_means
            targets_raw = batch.y.view(-1)

            # Calculate absolute errors
            abs_errors = torch.abs(preds_raw - targets_raw)

            # Accumulate per type
            # We move to CPU to accumulate in dictionary to avoid GPU sync overhead in loop
            abs_errors_np = abs_errors.cpu().numpy()
            type_indices_np = type_indices.cpu().numpy()

            for i in range(len(abs_errors_np)):
                t_idx = type_indices_np[i]
                err = abs_errors_np[i]
                t_name = type_names[t_idx]

                type_abs_error_sum[t_name] += err
                type_counts[t_name] += 1

    # Compute Final Metric
    # Log(MAE) for each type, then average
    log_maes = []
    print("Validation Metrics per Type:")
    for t_name in type_names:
        count = type_counts[t_name]
        if count > 0:
            mae = type_abs_error_sum[t_name] / count
            log_mae = np.log(mae)
            log_maes.append(log_mae)
            print(f"  {t_name}: MAE={mae:.6f}, LogMAE={log_mae:.6f}, Count={count}")
        else:
            print(f"  {t_name}: No samples in validation set.")

    if len(log_maes) > 0:
        final_metric = np.mean(log_maes)
    else:
        final_metric = 0.0

    return final_metric


def predict(model, loader, device, standardizer):
    """
    Generates predictions for the test set.
    Returns a DataFrame with 'id' and 'scalar_coupling_constant'.
    """
    model.eval()

    ids_list = []
    preds_list = []

    # Pre-load normalization stats
    means_tensor, stds_tensor = get_normalization_tensors(standardizer, device)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            preds_norm = model(batch).view(-1)

            # Inverse transform
            type_indices = batch.coupling_type.view(-1)
            batch_means = means_tensor[type_indices]
            batch_stds = stds_tensor[type_indices]

            preds_raw = preds_norm * batch_stds + batch_means

            # Collect results
            ids_list.extend(batch.id.view(-1).cpu().numpy())
            preds_list.extend(preds_raw.cpu().numpy())

    # Create DataFrame
    df_submission = pd.DataFrame(
        {"id": ids_list, "scalar_coupling_constant": preds_list}
    )

    return df_submission
