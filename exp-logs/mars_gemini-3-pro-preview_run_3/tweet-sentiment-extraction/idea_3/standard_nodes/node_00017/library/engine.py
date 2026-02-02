import os
import torch
import torch.nn as nn
import numpy as np
from library.utils import jaccard, decode_span


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(data_loader, model, optimizer, device, scheduler, criterion, config):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        start_positions = data["start_positions"].to(device)
        end_positions = data["end_positions"].to(device)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

        # Calculate loss
        loss = criterion(start_logits, start_positions) + criterion(
            end_logits, end_positions
        )

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, criterion):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            start_positions = data["start_positions"].to(device)
            end_positions = data["end_positions"].to(device)

            # Raw text data for Jaccard calculation
            texts = data["text"]
            selected_texts = data["selected_text"]
            offsets = data["offsets"].numpy()

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            # Calculate Loss
            loss = criterion(start_logits, start_positions) + criterion(
                end_logits, end_positions
            )
            losses.update(loss.item(), input_ids.size(0))

            # Decode Predictions
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(input_ids.size(0)):
                # Get best span indices (token level)
                idx_start, idx_end = decode_span(start_probs[i], end_probs[i])

                # Map token indices to character offsets to extract text
                # offsets[i] is shape (seq_len, 2) -> [[start, end], [start, end], ...]
                if idx_start < len(offsets[i]) and idx_end < len(offsets[i]):
                    char_start = offsets[i][idx_start][0]
                    char_end = offsets[i][idx_end][1]
                    pred_string = texts[i][char_start:char_end]
                else:
                    pred_string = ""

                # Calculate Jaccard Score
                score = jaccard(selected_texts[i], pred_string)
                jaccards.update(score, 1)

    return losses.avg, jaccards.avg


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    device,
    criterion,
    config,
    fold,
):
    """
    Main training loop for a specific fold.
    Handles epochs, logging, and early stopping.
    """
    best_jaccard = -1.0
    save_path = os.path.join(config.output_dir, f"model_fold_{fold}.bin")

    for epoch in range(config.epochs):
        train_loss = train_fn(
            train_loader, model, optimizer, device, scheduler, criterion, config
        )
        valid_loss, valid_jaccard = eval_fn(valid_loader, model, device, criterion)

        # Print full precision metrics
        print(f"Fold {fold} | Epoch {epoch + 1}/{config.epochs}")
        print(f"  Train Loss: {train_loss}")
        print(f"  Valid Loss: {valid_loss}")
        print(f"  Valid Jaccard: {valid_jaccard}")

        # Early Stopping: Save model only if validation Jaccard improves
        if valid_jaccard > best_jaccard:
            best_jaccard = valid_jaccard
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path}")

    return best_jaccard
