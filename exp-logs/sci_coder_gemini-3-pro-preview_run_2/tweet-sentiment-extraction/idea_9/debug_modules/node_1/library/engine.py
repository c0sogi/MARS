import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, jaccard
from library.config import Config


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the sum of CrossEntropyLoss for start and end logits with label smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes one training epoch using Mixed Precision (AMP).
    """
    model.train()
    losses = AverageMeter()

    # Initialize GradScaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    for step, data in enumerate(data_loader):
        # Move data to device
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        start_positions = data["start_tokens"].to(device, dtype=torch.long)
        end_positions = data["end_tokens"].to(device, dtype=torch.long)

        optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

        # Backward pass
        scaler.scale(loss).backward()

        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

        if step % Config.PRINT_FREQ == 0 or step == (len(data_loader) - 1):
            print(f"Train Step {step}/{len(data_loader)} | Loss: {losses.avg:.5f}")

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score using offset mapping for precise string extraction.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

            # Check if targets exist (validation vs test)
            if "start_tokens" in data and data["start_tokens"] is not None:
                start_positions = data["start_tokens"].to(device, dtype=torch.long)
                end_positions = data["end_tokens"].to(device, dtype=torch.long)
                calc_loss = True
            else:
                calc_loss = False

            # Forward pass
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            if calc_loss:
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = start_loss + end_loss
                losses.update(loss.item(), input_ids.size(0))

            # Convert logits to indices
            start_preds = torch.argmax(start_logits, dim=1).cpu().detach().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().detach().numpy()

            # Retrieve metadata for decoding
            texts = data["text"]
            selected_texts = data["selected_text"]
            sentiments = data["sentiment"]
            offsets = data["offsets"]

            for i in range(len(input_ids)):
                text = texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # Neutral Heuristic: Predict entire text if sentiment is neutral
                if sentiment == "neutral":
                    pred_text = text
                else:
                    idx_start = start_preds[i]
                    idx_end = end_preds[i]

                    # Enforce start <= end
                    if idx_end < idx_start:
                        idx_end = idx_start

                    # Decode using offsets
                    # Ensure indices are within bounds of the offset list
                    if idx_start >= len(offset):
                        idx_start = len(offset) - 1
                    if idx_end >= len(offset):
                        idx_end = len(offset) - 1

                    start_char = offset[idx_start][0]
                    end_char = offset[idx_end][1]

                    # Extract substring
                    pred_text = text[start_char:end_char]

                # Calculate Jaccard Score
                # selected_texts[i] might be empty if running on test set without targets,
                # but Jaccard handles it (returns 0 unless both empty).
                score = jaccard(pred_text, selected_texts[i])
                jaccards.update(score, 1)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Jaccard: {jaccards.avg}")

    return losses.avg, jaccards.avg
