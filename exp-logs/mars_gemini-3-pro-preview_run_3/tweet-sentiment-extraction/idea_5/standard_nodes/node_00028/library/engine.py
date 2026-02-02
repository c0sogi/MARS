import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import AverageMeter, jaccard


def loss_fn(start_logits, end_logits, start_labels, end_labels):
    """
    Computes the Loss for the architecture.

    Args:
        start_logits: (Batch, Seq_Len)
        end_logits: (Batch, Seq_Len)
        start_labels: (Batch,)
        end_labels: (Batch,)

    Returns:
        torch.Tensor: The scalar loss value.
    """
    # Using Label Smoothing to handle fuzzy boundaries
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_labels)
    end_loss = loss_fct(end_logits, end_labels)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        optimizer: PyTorch Optimizer
        device: torch.device
        scheduler: Learning rate scheduler (optional)

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()

    for batch in data_loader:
        # Move inputs and targets to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        with autocast():
            start_logits, end_logits = model(input_ids, attention_mask)

            # Compute Loss
            loss = loss_fn(
                start_logits,
                end_logits,
                start_labels,
                end_labels,
            )

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, tokenizer):
    """
    Evaluates the model on the validation set using Hybrid Decoding.

    Args:
        data_loader: PyTorch DataLoader
        model: The TweetModel
        device: torch.device
        tokenizer: Transformers tokenizer for decoding IDs to text

    Returns:
        tuple: (Average Loss, Average Jaccard Score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Compute Loss (for monitoring)
            loss = loss_fn(
                start_logits,
                end_logits,
                start_labels,
                end_labels,
            )
            losses.update(loss.item(), input_ids.size(0))

            # --- Summation Decoding Strategy ---

            # Get probabilities
            p_start = torch.softmax(start_logits, dim=1).cpu().numpy()
            p_end = torch.softmax(end_logits, dim=1).cpu().numpy()

            # Prepare data for decoding
            ids = input_ids.cpu().numpy()
            s_labels = start_labels.cpu().numpy()
            e_labels = end_labels.cpu().numpy()

            for i in range(len(ids)):
                # Per-sample probabilities
                ps = p_start[i]
                pe = p_end[i]

                best_score = -float("inf")
                best_start = 0
                best_end = 0

                seq_len = len(ps)

                # Search for best span (i, j) maximizing:
                # Score = P_start[i] + P_end[j]
                for s_idx in range(seq_len):
                    if ps[s_idx] < 0.001:
                        continue

                    for e_idx in range(s_idx, seq_len):
                        if pe[e_idx] < 0.001:
                            continue

                        score = ps[s_idx] + pe[e_idx]

                        if score > best_score:
                            best_score = score
                            best_start = s_idx
                            best_end = e_idx

                # Decode Prediction
                pred_ids = ids[i][best_start : best_end + 1]
                pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)

                # Decode Ground Truth
                target_ids = ids[i][s_labels[i] : e_labels[i] + 1]
                target_str = tokenizer.decode(target_ids, skip_special_tokens=True)

                # Calculate Jaccard
                score = jaccard(target_str, pred_str)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg
