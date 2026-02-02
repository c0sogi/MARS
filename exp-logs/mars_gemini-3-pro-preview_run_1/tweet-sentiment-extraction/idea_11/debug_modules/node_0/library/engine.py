import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard, get_best_start_end_idxs


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end logits against soft targets.

    Args:
        start_logits: (Batch, Seq)
        end_logits: (Batch, Seq)
        start_targets: (Batch, Seq) - Probability distribution
        end_targets: (Batch, Seq) - Probability distribution

    Returns:
        torch.Tensor: The average loss.
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # KLDivLoss expects input in log-probabilities
    start_loss = loss_fct(F.log_softmax(start_logits, dim=1), start_targets)
    end_loss = loss_fct(F.log_softmax(end_logits, dim=1), end_targets)

    return (start_loss + end_loss) / 2


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader.
        model: The neural network model.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_AMP)

    for data in data_loader:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        start_targets = data["start_targets"].to(device, dtype=torch.float)
        end_targets = data["end_targets"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        data_loader: PyTorch DataLoader.
        model: The neural network model.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (average_loss, average_jaccard_score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            start_targets = data["start_targets"].to(device, dtype=torch.float)
            end_targets = data["end_targets"].to(device, dtype=torch.float)

            offsets = data["offsets"].cpu().numpy()
            texts = data["text"]
            selected_texts = data["selected_text"]
            sentiments = data["sentiment"]

            # Forward pass
            with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                start_logits, end_logits = model(input_ids, attention_mask)
                loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

            losses.update(loss.item(), input_ids.size(0))

            # Decode predictions
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            for i in range(len(input_ids)):
                text = texts[i]
                sentiment = sentiments[i]
                selected_text = selected_texts[i]
                offset = offsets[i]

                # Logic: If neutral, return full text. Else, use model prediction.
                if sentiment == "neutral":
                    prediction = text
                else:
                    idx_start, idx_end = get_best_start_end_idxs(
                        start_logits[i], end_logits[i]
                    )

                    # Map token indices to character indices
                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    prediction = text[char_start:char_end]

                score = jaccard(selected_text, prediction)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg
