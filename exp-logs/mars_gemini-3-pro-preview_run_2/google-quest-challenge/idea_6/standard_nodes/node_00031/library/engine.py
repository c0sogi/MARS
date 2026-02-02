import torch
import torch.nn as nn
import numpy as np
from library.utils import compute_spearmanr
from library.config import Config


def train_one_epoch(
    model, loader, optimizer, device, criterion=None, max_grad_norm=1.0
):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        optimizer: Optimizer instance.
        device: Device to run training on (cpu or cuda).
        criterion: Loss function. Defaults to nn.BCELoss() if None.
        max_grad_norm: Maximum norm for gradient clipping.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    # Use BCELoss as the model outputs probabilities (Sigmoid activation)
    if criterion is None:
        criterion = nn.BCELoss()

    for batch in loader:
        # Unpack batch based on library/data.py structure
        # 0: qa_ids, 1: q_input_ids, 2: q_attention_mask, 3: q_token_type_ids,
        # 4: q_title_mask, 5: q_body_mask, 6: a_input_ids, 7: a_attention_mask,
        # 8: a_token_type_ids, 9: cat_feats, 10: targets

        # Move all batch elements to device
        batch = [b.to(device) for b in batch]

        q_input_ids = batch[1]
        q_attention_mask = batch[2]
        q_token_type_ids = batch[3]
        q_title_mask = batch[4]
        q_body_mask = batch[5]
        a_input_ids = batch[6]
        a_attention_mask = batch[7]
        a_token_type_ids = batch[8]
        cat_feats = batch[9]
        targets = batch[10]

        optimizer.zero_grad()

        # Forward pass
        preds = model(
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            q_title_mask,
            q_body_mask,
            a_input_ids,
            a_attention_mask,
            a_token_type_ids,
            cat_feats,
        )

        # Loss calculation
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device, criterion=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        device: Device to run evaluation on.
        criterion: Loss function. Defaults to nn.BCELoss() if None.

    Returns:
        tuple: (average_loss, spearman_correlation_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    if criterion is None:
        criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in loader:
            # Move all batch elements to device
            batch = [b.to(device) for b in batch]

            q_input_ids = batch[1]
            q_attention_mask = batch[2]
            q_token_type_ids = batch[3]
            q_title_mask = batch[4]
            q_body_mask = batch[5]
            a_input_ids = batch[6]
            a_attention_mask = batch[7]
            a_token_type_ids = batch[8]
            cat_feats = batch[9]
            targets = batch[10]

            # Forward pass
            preds = model(
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                q_title_mask,
                q_body_mask,
                a_input_ids,
                a_attention_mask,
                a_token_type_ids,
                cat_feats,
            )

            loss = criterion(preds, targets)
            running_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute metrics
    avg_loss = running_loss / len(loader)

    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        spearman_score = compute_spearmanr(all_preds, all_targets)
    else:
        spearman_score = 0.0

    return avg_loss, spearman_score
