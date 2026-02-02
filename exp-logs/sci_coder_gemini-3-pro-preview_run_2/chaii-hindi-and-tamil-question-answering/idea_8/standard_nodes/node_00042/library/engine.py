import torch
import torch.nn as nn
from library.config import Config
from library.utils import FGM


def loss_fn(
    start_logits,
    end_logits,
    answerable_logits,
    start_positions,
    end_positions,
    answerable_labels,
):
    """
    Computes the total loss as a sum of Span Loss (Start+End) and Answerability Loss.
    """
    # Span Loss (Cross Entropy)
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    span_loss = (start_loss + end_loss) / 2

    # Answerability Loss (Binary Cross Entropy)
    # answerable_logits: (B, 1), answerable_labels: (B)
    loss_fct_cls = nn.BCEWithLogitsLoss()
    cls_loss = loss_fct_cls(answerable_logits, answerable_labels.unsqueeze(1))

    # Total Loss
    total_loss = span_loss + cls_loss
    return total_loss


def get_optimizer_grouped_parameters(model, config):
    """
    Groups parameters for the optimizer to apply Layer-wise Learning Rate Decay (LLRD).
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # Base learning rate and decay factor
    init_lr = config.learning_rate
    decay_rate = config.llrd_decay
    weight_decay = config.weight_decay

    # Define parameter groups
    optimizer_grouped_parameters = []

    # Identify layers
    # XLM-R Large has 24 layers. We assume the backbone is named 'roberta'.
    # Structure: roberta.embeddings, roberta.encoder.layer.0 ... roberta.encoder.layer.23

    # 1. Capture Task Heads and Pooler (Top-most layers get full LR)
    head_params = []
    head_names = []
    for name, param in model.named_parameters():
        if "roberta.encoder.layer" not in name and "roberta.embeddings" not in name:
            head_params.append((name, param))
            head_names.append(name)

    # Group Head Params (Decay vs No Decay)
    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": init_lr,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": init_lr,
        }
    )

    # 2. Capture Backbone Layers (Apply LLRD)
    # We iterate from top layer (23) down to 0
    n_layers = 24  # XLM-R Large

    for layer_i in range(n_layers - 1, -1, -1):
        layer_lr = init_lr * (decay_rate ** (n_layers - 1 - layer_i))

        layer_params = []
        for name, param in model.named_parameters():
            if f"roberta.encoder.layer.{layer_i}." in name:
                layer_params.append((name, param))

        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in layer_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # 3. Capture Embeddings (Lowest LR)
    embedding_lr = init_lr * (decay_rate**n_layers)
    embedding_params = []
    for name, param in model.named_parameters():
        if "roberta.embeddings" in name:
            embedding_params.append((name, param))

    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in embedding_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        }
    )
    optimizer_grouped_parameters.append(
        {
            "params": [
                p for n, p in embedding_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": embedding_lr,
        }
    )

    return optimizer_grouped_parameters


def train_fn(data_loader, model, optimizer, device, scheduler, config):
    """
    Executes one training epoch with Adversarial Training (FGM) and Gradient Accumulation.
    """
    model.train()

    # Initialize FGM if enabled
    if config.use_fgm:
        fgm = FGM(model)

    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        answerable_label = batch["answerable_label"].to(device)

        # --- 1. Standard Forward Pass ---
        start_logits, end_logits, answerable_logits = model(input_ids, attention_mask)

        loss = loss_fn(
            start_logits,
            end_logits,
            answerable_logits,
            start_positions,
            end_positions,
            answerable_label,
        )

        # Scale loss for gradient accumulation
        loss = loss / config.accumulate_grad_batches
        loss.backward()

        # --- 2. Adversarial Training (FGM) ---
        if config.use_fgm:
            # Perturb embeddings based on gradients from the standard pass
            fgm.attack(epsilon=config.fgm_epsilon)

            # Forward pass on perturbed data
            start_logits_adv, end_logits_adv, answerable_logits_adv = model(
                input_ids, attention_mask
            )

            loss_adv = loss_fn(
                start_logits_adv,
                end_logits_adv,
                answerable_logits_adv,
                start_positions,
                end_positions,
                answerable_label,
            )

            # Scale adversarial loss
            loss_adv = loss_adv / config.accumulate_grad_batches

            # Accumulate adversarial gradients
            loss_adv.backward()

            # Restore original embeddings
            fgm.restore()

        # Track total loss (unscaled for logging)
        total_loss += loss.item() * config.accumulate_grad_batches

        # --- 3. Optimizer Step (Gradient Accumulation) ---
        if (step + 1) % config.accumulate_grad_batches == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    print(f"Training Loss: {avg_loss:.6f}")
    return avg_loss


def eval_fn(data_loader, model, device, config):
    """
    Executes the validation loop.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            answerable_label = batch["answerable_label"].to(device)

            start_logits, end_logits, answerable_logits = model(
                input_ids, attention_mask
            )

            loss = loss_fn(
                start_logits,
                end_logits,
                answerable_logits,
                start_positions,
                end_positions,
                answerable_label,
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss
