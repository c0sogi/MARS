import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end indices.

    Args:
        start_logits (torch.Tensor): Predicted start logits [Batch, SeqLen].
        end_logits (torch.Tensor): Predicted end logits [Batch, SeqLen].
        start_targets (torch.Tensor): Gaussian smoothed start targets [Batch, SeqLen].
        end_targets (torch.Tensor): Gaussian smoothed end targets [Batch, SeqLen].

    Returns:
        torch.Tensor: The average combined loss.
    """
    # KLDivLoss expects input to be log-probabilities
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    start_log_probs = torch.log_softmax(start_logits, dim=1)
    end_log_probs = torch.log_softmax(end_logits, dim=1)

    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    # Average the losses
    return 0.5 * start_loss + 0.5 * end_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        data_loader (DataLoader): The training dataloader.
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run on.
        scheduler (Scheduler, optional): The learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_targets = data["start_targets"].to(device)
        end_targets = data["end_targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask)

        # Calculate loss
        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Scheduler step (per batch)
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation or test set.

    Args:
        data_loader (DataLoader): The dataloader to evaluate.
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to run on.

    Returns:
        tuple: (Average Loss, Start Logits (np.array), End Logits (np.array))
    """
    model.eval()
    losses = AverageMeter()
    final_start_logits = []
    final_end_logits = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            start_targets = data["start_targets"].to(device)
            end_targets = data["end_targets"].to(device)

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Calculate loss
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            # Collect logits
            final_start_logits.append(start_logits.cpu().numpy())
            final_end_logits.append(end_logits.cpu().numpy())

    return (
        losses.avg,
        np.concatenate(final_start_logits),
        np.concatenate(final_end_logits),
    )
