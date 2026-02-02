import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import compute_spearmanr


def get_optimizer_params(model):
    """
    Sets up differential learning rates for Head and Backbone.
    """
    # Define head parameters explicitly based on model architecture
    head_params = (
        list(model.layer_norm.parameters())
        + list(model.head_proj.parameters())
        + list(model.head_final.parameters())
    )

    head_params_ids = list(map(id, head_params))

    # Backbone parameters are everything else
    backbone_params = [p for p in model.parameters() if id(p) not in head_params_ids]

    optimizer_parameters = [
        {
            "params": backbone_params,
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": head_params,
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    return optimizer_parameters


def get_scheduler(optimizer, num_train_steps):
    """
    Creates a linear schedule with warmup.
    """
    # We use a small warmup for the scheduler itself (e.g., 10% of steps or fixed)
    # Standard practice is often 0 or small warmup if we are doing Head Warmup via freezing.
    # Config doesn't specify scheduler warmup steps, assuming 0 or minimal.
    # We'll use 0.1 ratio or 0.
    num_warmup_steps = int(num_train_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch with Head Warmup and Gradient Accumulation.
    """
    model.train()

    # ==========================================
    # Head Warmup Strategy (Freezing Logic)
    # ==========================================
    freeze_backbone = epoch < Config.WARMUP_EPOCHS

    # Toggle gradients for backbone
    # Backbone components: embeddings, shared_layers, q_branch, a_branch
    backbone_modules = [
        model.embeddings,
        model.shared_layers,
        model.q_branch,
        model.a_branch,
    ]

    for module in backbone_modules:
        for param in module.parameters():
            param.requires_grad = not freeze_backbone

    # Ensure head is always trainable
    # Head components: layer_norm, head_proj, head_final
    head_modules = [model.layer_norm, model.head_proj, model.head_final]
    for module in head_modules:
        for param in module.parameters():
            param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids_q = data["input_ids_q"].to(device)
        attention_mask_q = data["attention_mask_q"].to(device)
        input_ids_a = data["input_ids_a"].to(device)
        attention_mask_a = data["attention_mask_a"].to(device)
        targets = data["targets"].to(device)

        batch_size = input_ids_q.size(0)

        # Forward
        logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
        loss = criterion(logits, targets)

        # Scale loss for accumulation
        loss = loss / Config.ACCUMULATION_STEPS
        loss.backward()

        running_loss += (loss.item() * Config.ACCUMULATION_STEPS) * batch_size
        dataset_size += batch_size

        # Optimization Step
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % Config.PRINT_FREQ == 0:
            print(
                f"Epoch {epoch+1} | Step {step+1} | Loss: {running_loss / dataset_size:.6f}"
            )

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.6f}")
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in dataloader:
            input_ids_q = data["input_ids_q"].to(device)
            attention_mask_q = data["attention_mask_q"].to(device)
            input_ids_a = data["input_ids_a"].to(device)
            attention_mask_a = data["attention_mask_a"].to(device)
            targets = data["targets"].to(device)

            batch_size = input_ids_q.size(0)

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Sigmoid for predictions
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Metric
    spearman_score = compute_spearmanr(all_preds, all_targets)

    print(f"Validation Loss: {val_loss}")  # Full precision
    print(f"Validation Spearman Correlation: {spearman_score}")  # Full precision

    return val_loss, spearman_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids_q = data["input_ids_q"].to(device)
            attention_mask_q = data["attention_mask_q"].to(device)
            input_ids_a = data["input_ids_a"].to(device)
            attention_mask_a = data["attention_mask_a"].to(device)

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
