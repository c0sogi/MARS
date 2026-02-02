import os
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from library.config import Config
from library.utils import AverageMeter, get_logger, quadratic_weighted_kappa
from library.awp import AWP


def get_optimizer_params(model, learning_rate, weight_decay, llrd_decay):
    """
    Constructs optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).
    """
    # Define parameter groups
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # 1. Identify layers
    # DeBERTa-v3 structure: backbone.embeddings, backbone.encoder.layer.{0..N}, pool, fc
    # If model is EssayModel, it has 'backbone', 'pool', 'fc'.
    # If model is AutoModelForMaskedLM, it has 'deberta' (backbone) and 'cls' (head).

    named_parameters = list(model.named_parameters())

    # Helper to determine layer ID for LLRD
    # Returns: (layer_index, is_head)
    # layer_index: 0 for embeddings, 1..N for encoder layers
    # is_head: True for pooler/regressor/cls head
    def get_layer_id(name):
        if "embeddings" in name:
            return 0, False
        elif "encoder.layer" in name:
            # Extract layer number
            # name format: backbone.encoder.layer.12.output...
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    return int(parts[i + 1]) + 1, False
            return 0, False  # Fallback
        else:
            return -1, True  # Head

    # Determine max layer index
    max_layer = 0
    for n, p in named_parameters:
        lid, is_head = get_layer_id(n)
        if not is_head and lid > max_layer:
            max_layer = lid

    # Assign LRs
    # Head gets learning_rate
    # Layer i gets learning_rate * (decay ** (max_layer - i))

    groups = {}

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        lid, is_head = get_layer_id(name)

        if is_head:
            lr = learning_rate
            group_name = "head"
        else:
            # Calculate decay
            # If max_layer is 24 (0..23 + embeddings),
            # Layer 24 (top) -> decay^0 = 1.0
            # Layer 1 (bottom) -> decay^23
            # Layer 0 (emb) -> decay^24
            # Note: lid is 1-based for encoder layers, 0 for embeddings
            # We want top encoder layer to be close to head LR
            distance = max_layer - lid
            lr = learning_rate * (llrd_decay**distance)
            group_name = f"layer_{lid}"

        weight_decay_val = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        if group_name not in groups:
            groups[group_name] = {
                "params": [],
                "weight_decay": weight_decay_val,
                "lr": lr,
            }
        else:
            # Ensure weight decay consistency within group if needed,
            # but usually we split by (decay / no decay) AND lr.
            # Simpler approach: Create unique key based on lr and weight_decay
            pass

    # Re-do grouping to handle weight decay properly
    # List of dicts
    final_groups = []

    # We iterate and create groups dynamically
    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        lid, is_head = get_layer_id(name)

        if is_head:
            lr = learning_rate
        else:
            distance = max_layer - lid
            lr = learning_rate * (llrd_decay**distance)

        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        final_groups.append({"params": [param], "weight_decay": wd, "lr": lr})

    return final_groups


def train_mlm(train_loader, valid_loader=None):
    """
    Executes the Masked Language Modeling (Stage 1) training loop.
    Saves the adapted backbone to Config.mlm_model_dir.
    """
    logger = get_logger()
    logger.info("Starting Stage 1: Domain-Adaptive Pre-training (MLM)...")

    device = Config.device

    # Initialize Model for MLM
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    model.to(device)
    model.train()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=Config.mlm_learning_rate,
        weight_decay=Config.mlm_weight_decay,
    )

    # Scheduler (Linear for MLM usually fine, or Cosine)
    num_training_steps = len(train_loader) * Config.mlm_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")

    for epoch in range(Config.mlm_epochs):
        model.train()
        losses = AverageMeter()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            losses.update(loss.item(), input_ids.size(0))

            if (step + 1) % Config.print_freq == 0:
                logger.info(
                    f"MLM Epoch {epoch+1}/{Config.mlm_epochs} [Step {step+1}/{len(train_loader)}] Loss: {losses.val:.4f} Avg: {losses.avg:.4f}"
                )

        logger.info(f"MLM Epoch {epoch+1} Completed. Average Loss: {losses.avg:.4f}")

        # Save checkpoint
        if losses.avg < best_loss:
            best_loss = losses.avg
            logger.info(
                f"Loss improved. Saving MLM checkpoint to {Config.mlm_model_dir}"
            )
            model.save_pretrained(Config.mlm_model_dir)

    # Clean up
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("MLM Training Finished.")


def train_fn(model, train_loader, optimizer, scheduler, epoch, awp=None):
    """
    Executes one epoch of Supervised Fine-Tuning (Stage 2).
    Includes Gradient Accumulation and Adversarial Weight Perturbation (AWP).
    """
    model.train()
    device = Config.device
    losses = AverageMeter()

    # For logging
    logger = get_logger()

    scaler = torch.cuda.amp.GradScaler()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            preds = model(input_ids, attention_mask)
            loss = nn.MSELoss()(preds, labels)

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # AWP Attack
        awp_triggered = False
        if awp is not None and epoch >= Config.awp_start_epoch:
            # Unscale gradients before AWP attack (needed for correct perturbation magnitude)
            scaler.unscale_(optimizer)
            awp_triggered = True

            # Attack: Perturb weights to maximize loss
            awp.attack()

            # Forward pass with perturbed weights
            with torch.cuda.amp.autocast():
                preds_adv = model(input_ids, attention_mask)
                loss_adv = nn.MSELoss()(preds_adv, labels)
                if Config.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / Config.gradient_accumulation_steps

            # Backward pass with perturbed weights
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp._restore()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            # Clip Gradients
            # Cite debug_lesson_4: Prevent GradScaler Double Unscaling
            if not awp_triggered:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics (scale loss back up for logging)
        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        if (step + 1) % Config.print_freq == 0:
            logger.info(
                f"Epoch {epoch+1} [Step {step+1}/{len(train_loader)}] Loss: {losses.val:.4f} Avg: {losses.avg:.4f}"
            )

    return losses.avg


def valid_fn(model, valid_loader):
    """
    Evaluates the model on the validation set.
    Returns Average Loss and Quadratic Weighted Kappa score.
    """
    model.eval()
    device = Config.device

    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            output = model(input_ids, attention_mask)

            loss = nn.MSELoss()(output, labels)
            losses.update(loss.item(), input_ids.size(0))

            # Collect predictions and targets
            preds.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    preds = torch.cat([torch.tensor(p) for p in preds]).numpy()  # Flatten
    targets = torch.cat([torch.tensor(t) for t in targets]).numpy()  # Flatten

    # Calculate Metric
    score = quadratic_weighted_kappa(targets, preds)

    return losses.avg, score


def inference_fn(model, test_loader):
    """
    Generates predictions for the test set.
    """
    model.eval()
    device = Config.device
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, attention_mask)
            preds.append(output.detach().cpu().numpy())

    preds = torch.cat([torch.tensor(p) for p in preds]).numpy()

    return preds
