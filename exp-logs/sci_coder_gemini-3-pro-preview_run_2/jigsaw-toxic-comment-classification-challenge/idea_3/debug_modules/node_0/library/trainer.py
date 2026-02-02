import os
import re
import time
import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, AdamW
import numpy as np

from library.config import Config
from library.utils import AverageMeter, compute_auc, save_checkpoint, seed_everything
from library.data_loader import get_dataloaders
from library.models import CustomTransformer


def get_llrd_parameters(model, base_lr, head_lr, weight_decay, decay_factor):
    """
    Sets up layer-wise learning rate decay for the model.

    Args:
        model: The PyTorch model.
        base_lr: The base learning rate for the top transformer layer.
        head_lr: The learning rate for the classification head.
        weight_decay: Weight decay coefficient.
        decay_factor: Multiplicative decay factor for lower layers.

    Returns:
        List of dictionaries containing parameter groups for the optimizer.
    """
    # Define parameter groups
    optimizer_parameters = []

    # Identify the classification head parameters
    # In CustomTransformer, the head is named 'head'
    head_params = list(map(id, model.head.parameters()))

    # Add head parameters with head_lr
    optimizer_parameters.append(
        {"params": model.head.parameters(), "lr": head_lr, "weight_decay": weight_decay}
    )

    # Determine the number of layers in the backbone
    # Typically 12 for base models
    if hasattr(model.model_config, "num_hidden_layers"):
        num_layers = model.model_config.num_hidden_layers
    elif hasattr(model.model_config, "n_layers"):
        num_layers = model.model_config.n_layers
    else:
        # Fallback if config doesn't specify clearly, though standard transformers do
        num_layers = 12

    # Iterate through the backbone parameters
    # We group them by layer index
    # Embeddings are treated as layer -1 (or effectively layer 0 in decay calculation)

    # We need to collect params by layer to assign LRs
    # Structure: {layer_idx: [params]}
    layer_params = {i: [] for i in range(num_layers)}
    embedding_params = []
    other_params = []  # LayerNorms, Poolers, etc. often interspersed

    for name, param in model.model.named_parameters():
        if id(param) in head_params:
            continue

        if not param.requires_grad:
            continue

        # Find layer index
        # Common patterns: encoder.layer.0, encoder.layers.0, h.0
        match = re.search(r"\.(?:layer|layers|h)\.(\d+)\.", name)

        if match:
            layer_idx = int(match.group(1))
            layer_params[layer_idx].append(param)
        elif "embedding" in name:
            embedding_params.append(param)
        else:
            # Usually LayerNorms at the start/end or poolers
            # We'll assign them the base_lr or a specific logic.
            # Often treating them as part of the top layer or separate high-level params is safe.
            other_params.append(param)

    # Assign LRs
    # Top layer (index num_layers - 1) gets base_lr
    # Layer i gets base_lr * (decay_factor ^ (num_layers - 1 - i))

    for i in range(num_layers - 1, -1, -1):
        lr = base_lr * (decay_factor ** (num_layers - 1 - i))
        if layer_params[i]:
            optimizer_parameters.append(
                {"params": layer_params[i], "lr": lr, "weight_decay": weight_decay}
            )

    # Embeddings get the lowest LR
    embed_lr = base_lr * (decay_factor**num_layers)
    if embedding_params:
        optimizer_parameters.append(
            {"params": embedding_params, "lr": embed_lr, "weight_decay": weight_decay}
        )

    # Other params (like final LayerNorm of the backbone)
    # We give them base_lr
    if other_params:
        optimizer_parameters.append(
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay}
        )

    return optimizer_parameters


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # We don't print progress bars, but we can print start status
    # print(f"Training Epoch {epoch+1}...")

    for step, batch in enumerate(train_loader):
        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        batch_size = ids.size(0)

        # Forward
        logits = model(ids, mask, token_type_ids)

        # Loss
        loss = criterion(logits, targets)

        # Update metrics
        losses.update(loss.item(), batch_size)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return losses.avg


def valid_fn(val_loader, model, criterion, device):
    """
    Validation loop.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    valid_labels = []

    # print("Validating...")

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            batch_size = ids.size(0)

            logits = model(ids, mask, token_type_ids)
            loss = criterion(logits, targets)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            preds.append(probs.detach().cpu().numpy())
            valid_labels.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    # Compute AUC
    score = compute_auc(valid_labels, preds)

    return losses.avg, score


def run_transformer_training(model_name, save_name):
    """
    Main function to train a transformer model.

    Args:
        model_name: HuggingFace model name (e.g., 'roberta-base').
        save_name: Filename to save the best model (e.g., 'best_model.bin').
    """
    seed_everything(Config.seed)
    device = Config.device

    print(f"Initializing model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # DataLoaders
    train_loader, val_loader = get_dataloaders(tokenizer)

    # Model
    model = CustomTransformer(model_name, config=Config)
    model.to(device)

    # Optimizer with LLRD
    optimizer_grouped_parameters = get_llrd_parameters(
        model,
        base_lr=Config.lr,
        head_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
        decay_factor=Config.llrd_decay,
    )

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=1e-6)

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_score = -np.inf
    patience_counter = 0
    best_loss = np.inf

    save_path = os.path.join(Config.working_dir, save_name)

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, device
        )

        # Validate
        avg_val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.0f}s")
        print(f"  Train Loss: {avg_train_loss}")
        print(f"  Val Loss: {avg_val_loss}")
        print(f"  Val AUC: {val_auc}")

        # Early Stopping & Checkpointing
        # We track best AUC
        if val_auc > best_score:
            best_score = val_auc
            print(f"  Validation AUC Improved. Saving model to {save_path}")
            save_checkpoint(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Training finished for {model_name}. Best AUC: {best_score}")
    return best_score
