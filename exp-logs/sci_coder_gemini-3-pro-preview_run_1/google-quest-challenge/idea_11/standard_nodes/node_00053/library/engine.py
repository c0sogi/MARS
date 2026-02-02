import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import AverageMeter, compute_spearman_correlation


def get_optimizer(model):
    """
    Configures the optimizer with differential learning rates and weight decay exclusions.

    Groups:
    1. Head, Fusion Norm, and Re-initialized Backbone Layers -> LR_HIGH
    2. Remaining Backbone Layers -> LR_LOW
    """
    # Define parameter groups
    high_lr_params = []
    low_lr_params = []

    # Identify re-initialized layers in the backbone
    # DistilRoBERTa structure: backbone.encoder.layer is a ModuleList
    # We need to identify which layers are re-initialized
    reinit_start_index = len(model.backbone.encoder.layer) - Config.REINIT_LAYERS

    # Iterate through named parameters to assign groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if parameter belongs to High LR group
        is_high_lr = False

        # 1. Residual Fusion Head
        if "head" in name:
            is_high_lr = True
        # 2. Fusion Normalization
        elif "fusion_norm" in name:
            is_high_lr = True
        # 3. Re-initialized Backbone Layers
        elif "backbone.encoder.layer" in name:
            # Extract layer index from name, e.g., "backbone.encoder.layer.4.attention..."
            try:
                parts = name.split(".")
                layer_idx = int(parts[3])  # backbone -> encoder -> layer -> idx
                if layer_idx >= reinit_start_index:
                    is_high_lr = True
            except (IndexError, ValueError):
                pass

        # Apply Weight Decay Exclusion (No WD for bias or LayerNorm)
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        weight_decay = (
            0.0 if any(nd in name for nd in no_decay) else Config.WEIGHT_DECAY
        )

        param_group = {
            "params": [param],
            "weight_decay": weight_decay,
            # LR will be assigned based on group
        }

        if is_high_lr:
            param_group["lr"] = Config.LR_HIGH
            high_lr_params.append(param_group)
        else:
            param_group["lr"] = Config.LR_LOW
            low_lr_params.append(param_group)

    # Create optimizer
    optimizer = torch.optim.AdamW(high_lr_params + low_lr_params)

    return optimizer


def get_scheduler(optimizer, num_train_steps):
    """
    Creates a linear schedule with warmup.
    """
    # 10% warmup is a safe default
    num_warmup_steps = int(0.1 * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )
    return scheduler


def train_fn(dataloader, model, optimizer, device, scheduler, epoch):
    """
    Training loop for one epoch. Handles progressive unfreezing.
    """
    model.train()
    loss_score = AverageMeter()
    loss_fn = nn.BCEWithLogitsLoss()

    # ==========================================
    # Progressive Unfreezing Strategy
    # ==========================================
    # Epoch 0 (1st epoch): Freeze lower backbone layers
    # Epoch 1+ (2nd epoch onwards): Unfreeze everything

    if epoch == 0:
        # Calculate split index
        total_layers = len(model.backbone.encoder.layer)
        reinit_start = total_layers - Config.REINIT_LAYERS

        # Freeze Embeddings
        for param in model.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze Lower Transformer Layers
        for i in range(reinit_start):
            for param in model.backbone.encoder.layer[i].parameters():
                param.requires_grad = False

        # Ensure Top Layers, Fusion, and Head are trainable
        for i in range(reinit_start, total_layers):
            for param in model.backbone.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in model.fusion_norm.parameters():
            param.requires_grad = True

        for param in model.head.parameters():
            param.requires_grad = True

        if Config.REINIT_LAYERS > 0:
            print(
                f"Epoch {epoch+1}: Lower backbone layers frozen. Training Head and Top {Config.REINIT_LAYERS} layers."
            )
        else:
            print(f"Epoch {epoch+1}: Backbone frozen. Training Head only.")

    else:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        if epoch == 1:
            print(f"Epoch {epoch+1}: Unfreezing entire model.")

    # ==========================================
    # Training Loop
    # ==========================================
    for step, data in enumerate(dataloader):
        # Move inputs to device
        q_input_ids = data["q_input_ids"].to(device)
        q_attention_mask = data["q_attention_mask"].to(device)
        a_input_ids = data["a_input_ids"].to(device)
        a_attention_mask = data["a_attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = q_input_ids.size(0)

        # Forward pass
        optimizer.zero_grad()

        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )

        # Compute Loss
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update metrics
        loss_score.update(loss.item(), batch_size)

    return loss_score.avg


def eval_fn(dataloader, model, device):
    """
    Validation loop. Computes Loss and Spearman Correlation.
    """
    model.eval()
    loss_score = AverageMeter()
    loss_fn = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for data in dataloader:
            q_input_ids = data["q_input_ids"].to(device)
            q_attention_mask = data["q_attention_mask"].to(device)
            a_input_ids = data["a_input_ids"].to(device)
            a_attention_mask = data["a_attention_mask"].to(device)
            labels = data["labels"].to(device)

            batch_size = q_input_ids.size(0)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            loss = loss_fn(logits, labels)
            loss_score.update(loss.item(), batch_size)

            # Apply sigmoid for predictions
            batch_preds = torch.sigmoid(logits).cpu().numpy()
            batch_targets = labels.cpu().numpy()

            preds.append(batch_preds)
            targets.append(batch_targets)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute Metric
    spearman_corr = compute_spearman_correlation(preds, targets)

    print(f"Validation Loss: {loss_score.avg}")
    print(f"Validation Spearman Correlation: {spearman_corr}")

    return loss_score.avg, spearman_corr


def inference_fn(dataloader, model, device):
    """
    Inference loop for test set. Returns probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            q_input_ids = data["q_input_ids"].to(device)
            q_attention_mask = data["q_attention_mask"].to(device)
            a_input_ids = data["a_input_ids"].to(device)
            a_attention_mask = data["a_attention_mask"].to(device)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            batch_preds = torch.sigmoid(logits).cpu().numpy()
            preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)
    return preds
