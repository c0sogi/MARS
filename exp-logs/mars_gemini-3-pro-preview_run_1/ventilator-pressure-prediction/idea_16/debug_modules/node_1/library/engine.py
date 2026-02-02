import torch
import numpy as np
from library.config import GRAD_CLIP


def train_one_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    """
    Performs one epoch of training using Composite Loss and OneCycleLR.

    Args:
        model (nn.Module): The VentilatorModel instance.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer (AdamW).
        scheduler (LRScheduler): The learning rate scheduler (OneCycleLR).
        device (torch.device): The computing device.
        loss_fn (nn.Module): Instance of MaskedL1Loss.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch in data_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        # Forward pass: Model returns (final_pred, aux_pred)
        pred, aux_pred = model(x)

        # Compute Composite Loss (Main + Weighted Aux)
        # MaskedL1Loss handles masking by u_out internally
        loss = loss_fn(pred, y, u_out, aux_pred)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step (OneCycleLR updates every batch)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(model, data_loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Computes MAE strictly on the inspiratory phase (u_out == 0).

    Args:
        model (nn.Module): The VentilatorModel instance.
        data_loader (DataLoader): DataLoader for validation data.
        device (torch.device): The computing device.
        loss_fn (nn.Module): Instance of MaskedL1Loss.

    Returns:
        float: The average MAE on the inspiratory phase.
    """
    model.eval()
    total_mae = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass: We only evaluate the main head
            pred, _ = model(x)

            # Compute Metric: Masked MAE
            # Passing aux_pred=None ensures only the main prediction error is calculated
            mae = loss_fn(pred, y, u_out, aux_pred=None)

            total_mae += mae.item()

    return total_mae / num_batches


def predict(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The VentilatorModel instance.
        data_loader (DataLoader): DataLoader for test data.
        device (torch.device): The computing device.

    Returns:
        tuple: (flat_ids, flat_preds) as numpy arrays.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            ids = batch["ids"]

            # Forward pass
            pred, _ = model(x)

            # Squeeze last dimension if necessary: (Batch, Seq, 1) -> (Batch, Seq)
            if pred.dim() == 3:
                pred = pred.squeeze(-1)

            all_preds.append(pred.cpu().numpy())
            all_ids.append(ids.numpy())

    # Flatten results to match submission format
    flat_preds = np.concatenate(all_preds).flatten()
    flat_ids = np.concatenate(all_ids).flatten()

    return flat_ids, flat_preds
