import torch
import torch.nn as nn
import numpy as np
from library.config import Config


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes a single training epoch using Mixed Precision and Gradient Accumulation.
    """
    model.train()
    total_loss = 0.0
    count = 0

    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Loss function for start and end token prediction
    loss_fct = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        start_positions = data["start_positions"].to(device, dtype=torch.long)
        end_positions = data["end_positions"].to(device, dtype=torch.long)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            start_logits, end_logits = model(input_ids, attention_mask=attention_mask)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2.0

            # Normalize loss for gradient accumulation
            if Config.GRAD_ACCUM_STEPS > 1:
                loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Track total loss (scale back to original magnitude for reporting)
        total_loss += loss.item() * Config.GRAD_ACCUM_STEPS
        count += 1

        # Optimizer step (with gradient accumulation)
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(data_loader):
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and scaler
            scaler.step(optimizer)
            scaler.update()

            # Step scheduler
            scheduler.step()

            # Zero gradients
            optimizer.zero_grad()

    avg_loss = total_loss / count
    return avg_loss


def eval_fn(data_loader, model, device):
    """
    Executes evaluation on the validation or test set.
    Returns the average loss (if targets exist) and the collected logits.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    all_start_logits = []
    all_end_logits = []

    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            # Use autocast for consistency and performance optimization
            with torch.cuda.amp.autocast():
                start_logits, end_logits = model(
                    input_ids, attention_mask=attention_mask
                )

            # Compute loss if targets are available (Validation mode)
            if "start_positions" in data and "end_positions" in data:
                start_positions = data["start_positions"].to(device, dtype=torch.long)
                end_positions = data["end_positions"].to(device, dtype=torch.long)

                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2.0

                total_loss += loss.item()
                count += 1

            # Collect logits (move to CPU, convert to float32 then numpy)
            all_start_logits.append(start_logits.float().detach().cpu().numpy())
            all_end_logits.append(end_logits.float().detach().cpu().numpy())

    # Concatenate all logits
    if len(all_start_logits) > 0:
        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits = np.concatenate(all_end_logits, axis=0)
    else:
        all_start_logits = np.array([])
        all_end_logits = np.array([])

    avg_loss = total_loss / count if count > 0 else 0.0

    return avg_loss, (all_start_logits, all_end_logits)
