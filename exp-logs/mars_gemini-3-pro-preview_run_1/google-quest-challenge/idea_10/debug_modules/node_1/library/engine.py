import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import AverageMeter, compute_spearman_correlation


def get_optimizer_params(model):
    """
    Configures the optimizer parameters with differential learning rates and weight decay.

    Groups:
    1. Backbone parameters: Low LR, Weight Decay (except bias/LayerNorm)
    2. Head/Fusion parameters: High LR, Weight Decay (except bias/LayerNorm)
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    for n, p in param_optimizer:
        if not p.requires_grad:
            continue

        # Determine Learning Rate
        # If the parameter belongs to the backbone, use the lower backbone LR.
        # Otherwise (head, fusion_norm), use the higher head LR.
        if "backbone" in n:
            lr = Config.LEARNING_RATE_BACKBONE
        else:
            lr = Config.LEARNING_RATE_HEAD

        # Determine Weight Decay
        # Exclude bias and LayerNorm parameters from weight decay.
        if any(nd in n for nd in no_decay):
            wd = 0.0
        else:
            wd = Config.WEIGHT_DECAY

        optimizer_parameters.append({"params": [p], "weight_decay": wd, "lr": lr})

    return optimizer_parameters


def train_fn(train_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids_q = batch["input_ids_q"].to(device)
        attention_mask_q = batch["attention_mask_q"].to(device)
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Forward pass
        logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Scheduler step (if per-step)
        if scheduler is not None:
            scheduler.step()

        # Zero gradients
        optimizer.zero_grad()

        # Update metrics
        losses.update(loss.item(), batch_size)

    return losses.avg


def eval_fn(val_loader, model, device):
    """
    Evaluates the model on the validation set.

    Returns:
        tuple: (average_loss, spearman_correlation)
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            batch_size = labels.size(0)

            # Forward pass
            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            loss = criterion(logits, labels)

            losses.update(loss.item(), batch_size)

            # Apply Sigmoid for predictions
            preds = torch.sigmoid(logits)

            preds_list.append(preds.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Compute metric
    score = compute_spearman_correlation(predictions, targets)

    return losses.avg, score


def predict_fn(test_loader, model, device):
    """
    Generates predictions for the test set.

    Returns:
        tuple: (predictions_numpy_array, list_of_qa_ids)
    """
    model.eval()
    preds_list = []
    qa_ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)

            if "qa_ids" in batch:
                qa_ids_list.extend(batch["qa_ids"])

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            preds = torch.sigmoid(logits)
            preds_list.append(preds.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)

    return predictions, qa_ids_list


def generate_submission(test_loader, model, device):
    """
    Generates predictions and saves them to the submission file in the correct format.
    """
    print("Generating predictions for test set...")
    predictions, qa_ids = predict_fn(test_loader, model, device)

    # Load sample submission to get column names
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    target_cols = [col for col in sample_sub.columns if col != "qa_id"]

    # Create submission DataFrame
    submission = pd.DataFrame(predictions, columns=target_cols)
    submission["qa_id"] = qa_ids

    # Reorder columns to match sample submission
    cols = ["qa_id"] + target_cols
    submission = submission[cols]

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
