import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Computes the total loss for start and end positions using CrossEntropyLoss
    with label smoothing.
    """
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def train_fn(dataloader, model, optimizer, device, scheduler, epoch, awp=None):
    """
    Executes one training epoch with Gradient Accumulation and Adversarial Weight Perturbation (AWP).
    """
    model.train()
    losses = AverageMeter()

    # Ensure gradients are zeroed before starting the loop
    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_positions = data["start_positions"].to(device)
        end_positions = data["end_positions"].to(device)

        batch_size = input_ids.size(0)

        # --- 1. Standard Forward Pass ---
        start_logits, end_logits = model(input_ids, attention_mask)
        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

        # Scale loss for gradient accumulation
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        # --- 2. Standard Backward Pass ---
        loss.backward()

        # --- 3. Adversarial Weight Perturbation (AWP) ---
        if awp is not None and awp.should_apply(epoch):
            # Perturb weights based on the gradients computed above
            awp.attack_step()

            # Forward pass with perturbed weights
            adv_start_logits, adv_end_logits = model(input_ids, attention_mask)
            adv_loss = loss_fn(
                adv_start_logits, adv_end_logits, start_positions, end_positions
            )

            # Scale adversarial loss
            if Config.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / Config.gradient_accumulation_steps

            # Accumulate gradients from adversarial loss
            adv_loss.backward()

            # Restore clean weights for the optimizer step
            awp.restore()

        # --- 4. Optimizer Step ---
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics (multiply back to get the actual loss value per batch)
        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

    return losses.avg


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the provided dataloader.
    Returns the average loss and the predictions (start/end logits).
    """
    model.eval()
    losses = AverageMeter()

    final_start_logits = []
    final_end_logits = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            # Compute loss if targets are available (Validation)
            if "start_positions" in data and "end_positions" in data:
                start_positions = data["start_positions"].to(device)
                end_positions = data["end_positions"].to(device)
                loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
                losses.update(loss.item(), input_ids.size(0))

            # Store logits on CPU to save GPU memory
            final_start_logits.append(start_logits.cpu().numpy())
            final_end_logits.append(end_logits.cpu().numpy())

    # Concatenate predictions from all batches
    if len(final_start_logits) > 0:
        predictions = (
            np.concatenate(final_start_logits),
            np.concatenate(final_end_logits),
        )
    else:
        predictions = (np.array([]), np.array([]))

    return losses.avg, predictions
