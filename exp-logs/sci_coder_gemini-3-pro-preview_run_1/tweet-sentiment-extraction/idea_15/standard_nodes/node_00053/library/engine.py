import numpy as np
import torch
import torch.nn as nn
from library.utils import AverageMeter
from library.config import Config


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end logits against soft targets.

    Args:
        start_logits: (batch_size, seq_len)
        end_logits: (batch_size, seq_len)
        start_targets: (batch_size, seq_len) - Gaussian smoothed probabilities
        end_targets: (batch_size, seq_len) - Gaussian smoothed probabilities
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # KLDivLoss expects log-probabilities as input
    start_log_probs = torch.log_softmax(start_logits, dim=1)
    end_log_probs = torch.log_softmax(end_logits, dim=1)

    # Compute loss for start and end
    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    # Return the average loss
    return (start_loss + end_loss) / 2


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        optimizer: PyTorch optimizer
        device: 'cuda' or 'cpu'
        scheduler: Learning rate scheduler

    Returns:
        Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    # Initialize Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    for batch in data_loader:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Scheduler Step (Update every batch)
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        device: 'cuda' or 'cpu'

    Returns:
        avg_loss: Average validation loss
        start_preds: Numpy array of start logits
        end_preds: Numpy array of end logits
    """
    model.eval()
    losses = AverageMeter()

    final_start_logits = []
    final_end_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_targets = batch.get("start_targets")
            end_targets = batch.get("end_targets")

            if start_targets is not None:
                start_targets = start_targets.to(device)
                end_targets = end_targets.to(device)

            # Forward pass (using autocast for consistency and efficiency on A100)
            with torch.cuda.amp.autocast():
                start_logits, end_logits = model(input_ids, attention_mask)

            # Calculate loss if targets are available
            if start_targets is not None:
                loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
                losses.update(loss.item(), input_ids.size(0))

            # Move logits to CPU and store as float32
            final_start_logits.append(start_logits.float().cpu().numpy())
            final_end_logits.append(end_logits.float().cpu().numpy())

    # Concatenate all batches
    start_preds = np.concatenate(final_start_logits, axis=0)
    end_preds = np.concatenate(final_end_logits, axis=0)

    return losses.avg, start_preds, end_preds
