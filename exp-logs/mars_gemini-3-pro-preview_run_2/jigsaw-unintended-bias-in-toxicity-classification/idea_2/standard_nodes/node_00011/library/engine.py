import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import JigsawMetrics


def loss_fn(outputs, targets, identities):
    """
    Computes the Aggressive Multi-Task Loss.

    Args:
        outputs (tuple): (toxicity_logits, identity_logits) from the model.
        targets (torch.Tensor): Toxicity labels (batch_size,).
        identities (torch.Tensor): Identity labels (batch_size, num_identities).

    Returns:
        torch.Tensor: The weighted combined loss.
    """
    toxicity_logits, identity_logits = outputs

    # Binary Cross Entropy for the main toxicity task
    # view(-1) ensures shape match if targets are (batch_size,) vs logits (batch_size, 1)
    tox_loss_fct = nn.BCEWithLogitsLoss()
    toxicity_loss = tox_loss_fct(toxicity_logits.view(-1), targets)

    # Binary Cross Entropy for the auxiliary identity task
    id_loss_fct = nn.BCEWithLogitsLoss()
    identity_loss = id_loss_fct(identity_logits, identities)

    # Weighted sum
    total_loss = (Config.TOXICITY_LOSS_WEIGHT * toxicity_loss) + (
        Config.IDENTITY_LOSS_WEIGHT * identity_loss
    )

    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Training loop for one epoch.

    Args:
        data_loader: PyTorch DataLoader for training data.
        model: The model to train.
        optimizer: Optimizer instance.
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    final_loss = 0

    for step, batch in enumerate(data_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        identities = batch["identities"].to(device, non_blocking=True)

        # --- Device-Side Trimming ---
        # Calculate the max length in this batch that is NOT padding
        # attention_mask is 1 for tokens, 0 for padding
        max_len = attention_mask.sum(dim=1).max().item()

        # Slice the tensors to this length to save compute
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        # ----------------------------

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = loss_fn(outputs, targets, identities)

        # Backward pass
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Update weights
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    Evaluation loop. Computes metrics on the validation set.

    Args:
        data_loader: PyTorch DataLoader for validation data.
        model: The model to evaluate.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    preds = []

    # No gradient needed for evaluation
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # --- Device-Side Trimming ---
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            # ----------------------------

            # Forward pass
            toxicity_logits, _ = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities
            batch_preds = torch.sigmoid(toxicity_logits).detach().cpu().numpy()
            preds.append(batch_preds)

    # Concatenate all predictions
    preds = np.concatenate(preds).flatten()

    # Load validation metadata to get ground truth labels and identities
    # We use the path from Config
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # In DEBUG mode, the dataloader is subsampled to 5000 rows, so we must slice the metadata to match
    if Config.DEBUG:
        val_df = val_df.iloc[:5000]

    # Compute metrics using the provided library tool
    metrics_tool = JigsawMetrics()
    metrics = metrics_tool.compute(val_df, preds)

    return metrics


def inference_fn(data_loader, model, device):
    """
    Inference loop for generating predictions on test set.

    Args:
        data_loader: PyTorch DataLoader for test data.
        model: The trained model.
        device: 'cuda' or 'cpu'.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # --- Device-Side Trimming ---
            max_len = attention_mask.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            # ----------------------------

            # Forward pass
            toxicity_logits, _ = model(input_ids, attention_mask)

            # Apply sigmoid
            batch_preds = torch.sigmoid(toxicity_logits).detach().cpu().numpy()
            preds.append(batch_preds)

    return np.concatenate(preds).flatten()
