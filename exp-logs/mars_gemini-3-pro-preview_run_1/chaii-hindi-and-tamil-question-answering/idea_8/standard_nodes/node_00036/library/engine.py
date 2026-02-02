import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from library.modeling import FGM


def get_optimizer(model, config):
    """
    Constructs the optimizer with differential learning rates and weight decay settings.
    Separates the backbone parameters from the head parameters using explicit module references.
    (Cite solution_lesson_node_00033)
    """
    no_decay = ["bias", "LayerNorm.weight"]

    def get_params(module, lr, weight_decay):
        decay = []
        no_decay_list = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_list.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_list, "weight_decay": 0.0, "lr": lr},
        ]

    optimizer_grouped_parameters = []

    # Backbone parameters
    optimizer_grouped_parameters.extend(
        get_params(model.backbone, config.lr_backbone, config.weight_decay)
    )

    # Head parameters (qa_outputs and relevance_classifier)
    optimizer_grouped_parameters.extend(
        get_params(model.qa_outputs, config.lr_head, config.weight_decay)
    )
    optimizer_grouped_parameters.extend(
        get_params(model.relevance_classifier, config.lr_head, config.weight_decay)
    )

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
    aux_weight=0.5,
):
    """
    Computes the combined loss: Span Loss (CrossEntropy) + Relevance Loss (BCE).
    Uses weighting for auxiliary task (Cite solution_lesson_node_00035).
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

    # Combine with weighting
    total_loss = span_loss + (aux_weight * rel_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler, config):
    """
    Executes one training epoch.
    """
    model.train()

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
            aux_weight=config.aux_loss_weight,
        )

        # --- 2. Standard Backward Pass ---
        loss.backward()

        # --- 3. Optimization Step ---
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    print(f"Training Loss: {avg_loss}")

    return avg_loss
