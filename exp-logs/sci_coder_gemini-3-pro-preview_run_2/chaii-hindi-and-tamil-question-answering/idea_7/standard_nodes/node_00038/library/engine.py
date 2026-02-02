import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter


def loss_fn(
    start_logits,
    end_logits,
    answerability_logits,
    start_positions,
    end_positions,
    answerable_labels,
):
    """
    Computes the combined loss for the Multi-Task QA model.

    L = (CrossEntropy(start) + CrossEntropy(end)) / 2 + weight * BCE(answerability)
    """
    # Span Loss
    loss_fct_span = nn.CrossEntropyLoss()
    start_loss = loss_fct_span(start_logits, start_positions)
    end_loss = loss_fct_span(end_logits, end_positions)
    span_loss = (start_loss + end_loss) / 2

    # Answerability Loss
    loss_fct_ans = nn.BCEWithLogitsLoss()
    ans_loss = loss_fct_ans(answerability_logits, answerable_labels)

    # Total Loss
    total_loss = span_loss + (Config.ANSWERABILITY_WEIGHT * ans_loss)
    return total_loss


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch):
    """
    Trains the model for one epoch with Gradient Accumulation.
    """
    model.train()
    losses = AverageMeter()

    # Initialize scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_FP16)

    # Zero gradients before starting the loop
    optimizer.zero_grad()

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        answerable_labels = batch["answerable_label"].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
            start_logits, end_logits, answerability_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss = loss_fn(
                start_logits,
                end_logits,
                answerability_logits,
                start_positions,
                end_positions,
                answerable_labels,
            )

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights only after accumulating enough gradients
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        batch_size = input_ids.size(0)
        # Log the actual loss (unscaled)
        losses.update(loss.item() * Config.GRAD_ACCUM_STEPS, batch_size)

    print(f"Epoch {epoch+1} Training Loss: {losses.avg}")
    return losses.avg


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Returns raw logits for post-processing and the average loss.
    """
    model.eval()
    losses = AverageMeter()

    # Lists to store predictions
    all_start_logits = []
    all_end_logits = []
    all_ans_logits = []

    # Check if dataloader provides labels (it does for val, might not for test)
    # We assume validation loader structure here based on QADataset

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            # We can use autocast during inference for speed/memory as well
            with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                start_logits, end_logits, answerability_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            # If labels are available, compute loss
            if "start_positions" in batch:
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                answerable_labels = batch["answerable_label"].to(device)

                loss = loss_fn(
                    start_logits,
                    end_logits,
                    answerability_logits,
                    start_positions,
                    end_positions,
                    answerable_labels,
                )
                losses.update(loss.item(), input_ids.size(0))

            # Store logits (move to CPU to save GPU memory)
            all_start_logits.append(start_logits.float().cpu().numpy())
            all_end_logits.append(end_logits.float().cpu().numpy())
            all_ans_logits.append(answerability_logits.float().cpu().numpy())

    # Concatenate all batches
    if len(all_start_logits) > 0:
        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits = np.concatenate(all_end_logits, axis=0)
        all_ans_logits = np.concatenate(all_ans_logits, axis=0)
    else:
        all_start_logits = np.array([])
        all_end_logits = np.array([])
        all_ans_logits = np.array([])

    if losses.count > 0:
        print(f"Validation Loss: {losses.avg}")

    return {
        "start_logits": all_start_logits,
        "end_logits": all_end_logits,
        "answerability_logits": all_ans_logits,
        "loss": losses.avg if losses.count > 0 else None,
    }
