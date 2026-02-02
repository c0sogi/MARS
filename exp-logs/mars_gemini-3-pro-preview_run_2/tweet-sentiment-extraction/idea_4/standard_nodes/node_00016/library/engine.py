import torch
import torch.nn as nn
import numpy as np
from library.config import Config


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the CrossEntropyLoss with label smoothing for start and end token predictions.

    Args:
        start_logits (torch.Tensor): Predicted logits for start position.
        end_logits (torch.Tensor): Predicted logits for end position.
        start_positions (torch.Tensor): Ground truth start indices.
        end_positions (torch.Tensor): Ground truth end indices.

    Returns:
        torch.Tensor: The sum of the start and end losses.
    """
    # Using PyTorch's built-in label smoothing in CrossEntropyLoss
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)

    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        data_loader (DataLoader): The training dataloader.
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer.
        device (str): Device to run training on.
        scheduler (LRScheduler, optional): The learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0

    for data in data_loader:
        # Move batch data to device
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        start_positions = data["start_tokens"].to(device)
        end_positions = data["end_tokens"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Calculate loss
        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Scheduler step (update learning rate per batch)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        data_loader (DataLoader): The validation dataloader.
        model (nn.Module): The model to evaluate.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (start_preds, end_preds, avg_loss)
            start_preds (np.ndarray): Concatenated start logits.
            end_preds (np.ndarray): Concatenated end logits.
            avg_loss (float): Average loss over the validation set.
    """
    model.eval()
    total_loss = 0
    start_preds = []
    end_preds = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            start_positions = data["start_tokens"].to(device)
            end_positions = data["end_tokens"].to(device)

            # Forward pass
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Calculate loss
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            total_loss += loss.item()

            # Store logits (moved to CPU)
            start_preds.append(start_logits.detach().cpu().numpy())
            end_preds.append(end_logits.detach().cpu().numpy())

    # Concatenate predictions from all batches
    start_preds = np.concatenate(start_preds)
    end_preds = np.concatenate(end_preds)

    return start_preds, end_preds, total_loss / len(data_loader)
