import torch
import numpy as np
from library.utils import MaskedMAELoss


def train_fn(model, data_loader, optimizer, device, loss_fn, max_grad_norm):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the training on (CPU/GPU).
        loss_fn (nn.Module): The loss function (MaskedMAELoss).
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_idx, (inputs, targets, u_out) in enumerate(data_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        u_out = u_out.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fn(outputs, targets, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / num_samples


def eval_fn(model, data_loader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the evaluation on.
        loss_fn (nn.Module): The loss function (MaskedMAELoss).

    Returns:
        float: Average loss on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, u_out) in enumerate(data_loader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            u_out = u_out.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_fn(outputs, targets, u_out)

            # Accumulate loss
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

    return running_loss / num_samples


def predict_fn(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The trained PyTorch model.
        data_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Flattened array of predictions matching the submission format.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(data_loader):
            # Move inputs to device
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Flatten the output (Batch, 80) -> (Batch * 80)
            # and move to CPU numpy
            preds_flat = outputs.cpu().numpy().flatten()
            predictions.append(preds_flat)

    # Concatenate all batches into a single array
    return np.concatenate(predictions)
