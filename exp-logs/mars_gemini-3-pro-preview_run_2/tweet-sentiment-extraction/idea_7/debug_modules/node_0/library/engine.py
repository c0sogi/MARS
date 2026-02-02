import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard
from library.loss import TweetLoss


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch using Mixed Precision.

    Args:
        data_loader: PyTorch DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        device: The computing device (CPU/GPU).
        scheduler: Learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    criterion = TweetLoss()

    for data in data_loader:
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        start_targets = data["start_targets"].to(device, dtype=torch.long)
        end_targets = data["end_targets"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            start_logits, end_logits = model(ids, mask, token_type_ids)
            loss = criterion(start_logits, end_logits, start_targets, end_targets)

        # Backward Pass and Optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        losses.update(loss.item(), ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Computes the Hybrid Loss and the Word-Level Jaccard Score.

    Args:
        data_loader: PyTorch DataLoader for validation data.
        model: The neural network model.
        device: The computing device.

    Returns:
        tuple: (average_loss, average_jaccard_score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    criterion = TweetLoss()

    with torch.no_grad():
        for data in data_loader:
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            start_targets = data["start_targets"].to(device, dtype=torch.long)
            end_targets = data["end_targets"].to(device, dtype=torch.long)

            # Metadata required for Jaccard score reconstruction
            orig_texts = data["orig_text"]
            sentiments = data["sentiment"]
            offsets = data["offsets"].cpu().numpy()

            # Forward Pass
            start_logits, end_logits = model(ids, mask, token_type_ids)
            loss = criterion(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), ids.size(0))

            # Convert logits to probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            start_targets_np = start_targets.cpu().detach().numpy()
            end_targets_np = end_targets.cpu().detach().numpy()

            # Iterate over the batch to reconstruct strings and compute Jaccard
            for i in range(ids.size(0)):
                text = orig_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # 1. Reconstruct Ground Truth String from Targets
                # We extract the substring using the target token indices and offsets
                gt_start_idx = start_targets_np[i]
                gt_end_idx = end_targets_np[i]

                if gt_start_idx < len(offset) and gt_end_idx < len(offset):
                    gt_char_start = offset[gt_start_idx][0]
                    gt_char_end = offset[gt_end_idx][1]
                    target_text = text[gt_char_start:gt_char_end]
                else:
                    # Fallback if indices are out of bounds (rare)
                    target_text = text

                # 2. Reconstruct Predicted String from Logits
                pred_start_idx = np.argmax(start_probs[i])
                pred_end_idx = np.argmax(end_probs[i])

                # Enforce start <= end constraint
                if pred_start_idx > pred_end_idx:
                    pred_end_idx = pred_start_idx

                # Apply Neutral Heuristic: If sentiment is neutral, predict full text
                if sentiment == "neutral":
                    pred_text = text
                else:
                    if pred_start_idx < len(offset) and pred_end_idx < len(offset):
                        pred_char_start = offset[pred_start_idx][0]
                        pred_char_end = offset[pred_end_idx][1]
                        pred_text = text[pred_char_start:pred_char_end]
                    else:
                        pred_text = text

                # 3. Compute Jaccard Score
                score = jaccard(pred_text, target_text)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg
