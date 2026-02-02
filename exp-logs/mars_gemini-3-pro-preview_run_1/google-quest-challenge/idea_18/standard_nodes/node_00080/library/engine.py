import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_spearmanr


def train_fn(model, dataloader, optimizer, scheduler, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader containing training data.
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler.
        device: The device (cpu/cuda) to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    final_loss = 0
    count = 0

    # Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()

    accumulation_steps = Config.ACCUMULATION_STEPS

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids_q = batch["input_ids_q"].to(device)
        attention_mask_q = batch["attention_mask_q"].to(device)
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

        # Calculate loss
        loss = criterion(logits, labels)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Track scaled-back loss for reporting
        final_loss += loss.item() * accumulation_steps
        count += 1

        # Perform optimization step if accumulation is complete
        if (step + 1) % accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

    avg_loss = final_loss / count
    print(f"Training Loss: {avg_loss}")

    return avg_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on validation or test data.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader containing val/test data.
        device: The device (cpu/cuda) to run on.

    Returns:
        tuple: (score, predictions)
            score (float): Spearman's correlation (0.0 if no labels provided).
            predictions (np.ndarray): Array of predicted probabilities.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)

            # Forward pass
            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

            # Apply Sigmoid activation to get probabilities [0, 1]
            preds = torch.sigmoid(logits)

            preds_list.append(preds.cpu().numpy())

            # Collect targets if they exist (Validation mode)
            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)

    score = 0.0
    if len(targets_list) > 0:
        targets = np.concatenate(targets_list, axis=0)
        score = compute_spearmanr(predictions, targets)
        # Print full precision as requested
        print(f"Validation Spearman Correlation: {score}")

    return score, predictions
