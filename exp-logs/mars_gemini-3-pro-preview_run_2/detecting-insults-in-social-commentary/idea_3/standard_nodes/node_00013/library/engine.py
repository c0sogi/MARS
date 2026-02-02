import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_auc


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures layer-wise learning rate decay (LLRD) for the model.

    Args:
        model (nn.Module): The model to optimize.
        encoder_lr (float): Base learning rate for the encoder (top layer).
        decoder_lr (float): Learning rate for the classification head.
        weight_decay (float): Weight decay coefficient.

    Returns:
        list: List of parameter groups for the optimizer.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # DeBERTa-v3-base has 12 layers.
    # Structure: backbone.embeddings, backbone.encoder.layer.0 ... .11, fc
    num_layers = 12

    # Initialize groups:
    # 0: Embeddings
    # 1..12: Encoder Layers 0..11
    # 13: Head/Decoder
    groups = [[] for _ in range(num_layers + 2)]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone.embeddings" in name:
            groups[0].append((name, param))
        elif "backbone.encoder.layer" in name:
            # Extract layer index from name (e.g., backbone.encoder.layer.5.output...)
            try:
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                groups[layer_idx + 1].append((name, param))
            except (ValueError, IndexError):
                # Fallback to embeddings group if parsing fails
                groups[0].append((name, param))
        elif "fc" in name:
            groups[num_layers + 1].append((name, param))
        else:
            # Other backbone params (e.g. final LayerNorm, pooler) go to the top encoder layer group
            groups[num_layers].append((name, param))

    optimizer_parameters = []

    # 1. Add Head Parameters
    for name, p in groups[num_layers + 1]:
        optimizer_parameters.append(
            {
                "params": [p],
                "weight_decay": (
                    0.0 if any(nd in name for nd in no_decay) else weight_decay
                ),
                "lr": decoder_lr,
            }
        )

    # 2. Add Backbone Parameters with Decay
    # Layer 11 (index 12) gets encoder_lr
    # Layer 0 (index 1) gets encoder_lr * (decay ** 11)
    # Embeddings (index 0) gets encoder_lr * (decay ** 12)

    for i in range(num_layers + 1):
        # Calculate decay exponent based on distance from top
        # i=12 (top layer) -> distance 0
        # i=0 (embeddings) -> distance 12
        distance = num_layers - i
        lr = encoder_lr * (Config.llrd_decay**distance)

        for name, p in groups[i]:
            optimizer_parameters.append(
                {
                    "params": [p],
                    "weight_decay": (
                        0.0 if any(nd in name for nd in no_decay) else weight_decay
                    ),
                    "lr": lr,
                }
            )

    return optimizer_parameters


def train_fn(train_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Device to run on.
        scheduler (Scheduler, optional): Learning rate scheduler.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Update weights
        optimizer.step()

        # Update scheduler (usually per step for transformers)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on validation or test data.

    Args:
        data_loader (DataLoader): DataLoader for validation/test data.
        model (nn.Module): The model.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    final_preds = []
    final_targets = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Calculate loss if labels are present
            if "labels" in batch:
                labels = batch["labels"].to(device)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                final_targets.extend(labels.cpu().numpy())

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)
            final_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    return avg_loss, np.array(final_preds), np.array(final_targets)
