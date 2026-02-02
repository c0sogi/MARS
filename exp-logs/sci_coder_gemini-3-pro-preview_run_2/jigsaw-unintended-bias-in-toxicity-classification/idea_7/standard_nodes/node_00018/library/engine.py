import torch
import numpy as np
from library.config import Config
from library.metrics import JigsawEvaluator


def train_one_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Torch device (cuda/cpu).
        loss_fn: Instance of HybridContrastiveLoss.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        identities = batch["identities"].to(device, non_blocking=True)

        # Device-Side Trimming
        # Calculate the maximum sequence length in this batch (ignoring padding)
        # We assume standard right-padding where attention_mask is 1 for tokens and 0 for pads
        max_len = int(torch.max(torch.sum(attention_mask, dim=1)).item())
        # Safety check to ensure length is at least 1
        max_len = max(max_len, 1)

        # Slice inputs to the effective length
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        optimizer.zero_grad()

        # Forward pass
        # model returns (toxicity_logit, identity_logits)
        tox_logits, ident_logits = model(input_ids, attention_mask)

        # Calculate Hybrid Loss
        loss = loss_fn(tox_logits, ident_logits, targets, identities)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization steps
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def valid_fn(model, data_loader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for validation data.
        device: Torch device.
        loss_fn: Loss function instance.

    Returns:
        tuple: (Average Loss, Metrics Dictionary)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    evaluator = JigsawEvaluator()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            identities = batch["identities"].to(device, non_blocking=True)

            # Device-Side Trimming
            max_len = int(torch.max(torch.sum(attention_mask, dim=1)).item())
            max_len = max(max_len, 1)

            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            # Forward pass
            tox_logits, ident_logits = model(input_ids, attention_mask)

            # Calculate Loss
            loss = loss_fn(tox_logits, ident_logits, targets, identities)

            total_loss += loss.item()
            num_batches += 1

            # Update evaluator with batch results
            evaluator.update(tox_logits, targets, identities)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute competition metrics
    metrics = evaluator.compute()

    return avg_loss, metrics


def inference_fn(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for test data.
        device: Torch device.

    Returns:
        np.ndarray: Flattened array of toxicity probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Device-Side Trimming
            max_len = int(torch.max(torch.sum(attention_mask, dim=1)).item())
            max_len = max(max_len, 1)

            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

            # Forward pass (we only need toxicity logits for inference)
            tox_logits, _ = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities [0, 1]
            preds = torch.sigmoid(tox_logits).detach().cpu().numpy().reshape(-1)
            all_preds.append(preds)

    return np.concatenate(all_preds)
