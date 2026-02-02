import torch
import torch.nn as nn
import numpy as np
from library.utils import compute_score
from library.config import Config


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, config: Config
):
    """
    Trains the model for one epoch using Mixed Precision and Gradient Accumulation.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: The training dataloader.
        device: The device to run training on.
        epoch: Current epoch number.
        config: Configuration object.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    dataset_size = 0
    running_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            # Scale loss for gradient accumulation
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track Loss (multiply by accumulation steps to get back original scale for logging)
        # Note: 'loss' variable is already scaled down, so we multiply back
        current_loss = loss.item() * config.gradient_accumulation_steps
        running_loss += current_loss * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch+1} Train Loss: {epoch_loss}")

    return epoch_loss


def validate(model, dataloader, device, config: Config):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The device to run evaluation on.
        config: Configuration object.

    Returns:
        tuple: (Average Validation Loss, Pearson Correlation Score)
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            # Forward pass (Autocast optional for inference but good for consistency/memory)
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

            loss = outputs.loss
            logits = outputs.logits

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions and targets for metric computation
            # Logits shape: [batch_size, 1] -> flatten to [batch_size]
            preds.append(logits.view(-1).cpu().numpy())
            targets.append(labels.view(-1).cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute Pearson Correlation
    score = compute_score(targets, preds)

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Pearson Score: {score}")

    return val_loss, score
