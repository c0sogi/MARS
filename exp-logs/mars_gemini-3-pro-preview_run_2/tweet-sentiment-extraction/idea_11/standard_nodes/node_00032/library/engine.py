import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, jaccard
from library.config import Config


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the sum of Cross Entropy Loss for start and end logits
    with Label Smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    # Initialize GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_FP16)

    for batch in data_loader:
        # Move batch data to the computation device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        optimizer.zero_grad()

        # Forward pass with Automatic Mixed Precision
        with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)

        # Update loss meter
        losses.update(loss.item(), input_ids.size(0))

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.
    Returns: avg_loss, avg_jaccard, start_logits, end_logits
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # Containers for logits
    final_start_logits = []
    final_end_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_targets = batch["start_targets"].to(device)
            end_targets = batch["end_targets"].to(device)

            # Metadata for Jaccard calculation
            texts = batch["text"]
            selected_texts = batch["selected_text"]
            sentiments = batch["sentiment"]
            offsets = batch["offsets"].cpu().numpy()

            # Forward pass
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            # Store logits (move to CPU/Numpy for ensemble/return)
            start_preds_np = start_logits.cpu().detach().numpy()
            end_preds_np = end_logits.cpu().detach().numpy()

            final_start_logits.append(start_preds_np)
            final_end_logits.append(end_preds_np)

            # Calculate Jaccard Score for monitoring
            # Apply softmax for local decoding
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            start_indices = np.argmax(start_probs, axis=1)
            end_indices = np.argmax(end_probs, axis=1)

            for i in range(len(input_ids)):
                text = str(texts[i])
                selected_text = str(selected_texts[i])
                sentiment = str(sentiments[i])
                offset = offsets[i]

                pred_selected_text = ""

                # Neutral Heuristic: If sentiment is neutral, predict the whole text
                if sentiment == "neutral":
                    pred_selected_text = text
                else:
                    idx_start = start_indices[i]
                    idx_end = end_indices[i]

                    if idx_end < idx_start:
                        idx_end = idx_start

                    # Extract text using offsets
                    # Ensure indices are valid within the offset array
                    if idx_start < len(offset) and idx_end < len(offset):
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]

                        # Handle padding or invalid tokens (0,0 is usually padding/special)
                        if char_start == 0 and char_end == 0 and idx_start != 0:
                            pred_selected_text = text
                        else:
                            pred_selected_text = text[char_start:char_end]
                    else:
                        pred_selected_text = text

                # Compute Jaccard
                score = jaccard(pred_selected_text, selected_text)
                jaccards.update(score, 1)

    return (
        losses.avg,
        jaccards.avg,
        np.concatenate(final_start_logits),
        np.concatenate(final_end_logits),
    )
