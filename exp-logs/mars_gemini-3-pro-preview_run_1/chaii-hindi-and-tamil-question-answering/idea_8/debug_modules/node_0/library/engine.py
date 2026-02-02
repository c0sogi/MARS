import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from library.modeling import FGM


def get_optimizer(model, config):
    """
    Constructs the optimizer with differential learning rates and weight decay settings.
    Separates the backbone parameters from the head parameters.
    """
    # Define parameter groups
    # We distinguish between the pre-trained backbone and the task-specific heads
    # to apply different learning rates.

    no_decay = ["bias", "LayerNorm.weight"]

    # Storage for parameter groups
    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter belongs to the backbone
        if "backbone" in name:
            if any(nd in name for nd in no_decay):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            # Parameters for qa_outputs and relevance_classifier
            if any(nd in name for nd in no_decay):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    optimizer_grouped_parameters = [
        {
            "params": backbone_decay,
            "weight_decay": config.weight_decay,
            "lr": config.lr_backbone,
        },
        {
            "params": backbone_no_decay,
            "weight_decay": 0.0,
            "lr": config.lr_backbone,
        },
        {
            "params": head_decay,
            "weight_decay": config.weight_decay,
            "lr": config.lr_head,
        },
        {
            "params": head_no_decay,
            "weight_decay": 0.0,
            "lr": config.lr_head,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=config.adam_epsilon)

    return optimizer


def get_scheduler(optimizer, num_train_steps, config):
    """
    Creates a linear schedule with warmup.
    """
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )
    return scheduler


def loss_fn(
    start_logits,
    end_logits,
    relevance_logits,
    start_positions,
    end_positions,
    relevance_labels,
):
    """
    Computes the combined loss: Span Loss (CrossEntropy) + Relevance Loss (BCE).
    """
    # Span Loss
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    span_loss = (start_loss + end_loss) / 2

    # Relevance Loss
    # relevance_logits: (Batch,), relevance_labels: (Batch,)
    rel_loss_fct = nn.BCEWithLogitsLoss()
    rel_loss = rel_loss_fct(relevance_logits, relevance_labels)

    # Combine
    total_loss = span_loss + rel_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler, config):
    """
    Executes one training epoch.
    Includes Adversarial Training (FGM) logic.
    """
    model.train()

    # Initialize FGM if enabled
    fgm = None
    if config.use_fgm:
        fgm = FGM(model)

    total_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        # Move inputs to device
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        start_positions = data["start_positions"].to(device)
        end_positions = data["end_positions"].to(device)
        relevance_labels = data["relevance_labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # --- 1. Standard Forward Pass ---
        start_logits, end_logits, relevance_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        loss = loss_fn(
            start_logits,
            end_logits,
            relevance_logits,
            start_positions,
            end_positions,
            relevance_labels,
        )

        # --- 2. Standard Backward Pass ---
        loss.backward()

        # --- 3. Adversarial Training (FGM) ---
        if config.use_fgm:
            # Attack: Perturb embeddings based on gradients
            fgm.attack(epsilon=config.fgm_epsilon, emb_name=config.fgm_param_name)

            # Forward pass with perturbed embeddings
            start_logits_adv, end_logits_adv, relevance_logits_adv = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss_adv = loss_fn(
                start_logits_adv,
                end_logits_adv,
                relevance_logits_adv,
                start_positions,
                end_positions,
                relevance_labels,
            )

            # Backward pass on adversarial loss (accumulate gradients)
            loss_adv.backward()

            # Restore original embeddings
            fgm.restore(emb_name=config.fgm_param_name)

        # --- 4. Optimization Step ---
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    print(f"Training Loss: {avg_loss}")

    return avg_loss
