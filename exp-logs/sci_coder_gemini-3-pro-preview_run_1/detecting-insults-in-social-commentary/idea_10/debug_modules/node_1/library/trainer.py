import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, get_score, save_checkpoint, AWP
from library.model import HybridDeberta


def train_fn(
    train_loader, model, criterion, optimizer, scheduler, device, epoch, awp=None
):
    """
    Training loop for one epoch.
    Includes Adversarial Weight Perturbation (AWP) logic.
    """
    model.train()
    losses = AverageMeter()

    # Determine if AWP should be applied in this epoch
    use_awp = False
    if awp is not None and epoch >= Config.AWP_START_EPOCH:
        use_awp = True

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        labels = batch["label"].to(device)

        batch_size = labels.size(0)

        # --- Standard Step ---
        # Forward pass
        y_preds = model(input_ids, attention_mask, svd_features)

        # Calculate loss (BCEWithLogitsLoss handles logits -> probabilities)
        loss = criterion(y_preds.view(-1), labels)

        # Backward pass to compute gradients
        loss.backward()

        # --- AWP Step ---
        if use_awp:
            # Perturb weights based on gradients
            awp.attack()

            # Re-compute forward pass and loss with perturbed weights
            y_preds_adv = model(input_ids, attention_mask, svd_features)
            loss_adv = criterion(y_preds_adv.view(-1), labels)

            # Accumulate gradients from adversarial loss
            loss_adv.backward()

            # Restore original weights
            awp.restore()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        losses.update(loss.item(), batch_size)

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    """
    Validation loop. Returns average loss and predictions.
    """
    model.eval()
    losses = AverageMeter()
    preds = []

    for step, batch in enumerate(valid_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        labels = batch["label"].to(device)

        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, svd_features)
            loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), batch_size)
        # Apply sigmoid to convert logits to probabilities
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    """
    Inference loop for generating predictions on test data (no labels).
    """
    model.eval()
    preds = []

    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)

        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, svd_features)

        preds.append(y_preds.sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions


def run_fold(fold, train_loader, valid_loader, device, save_path):
    """
    Orchestrates the training process for a single fold.
    """
    print(f"Training Fold: {fold}")

    # Initialize Model
    model = HybridDeberta(pretrained=True)
    model.to(device)

    # --- Optimizer with Differential Learning Rates ---
    # Separate backbone parameters from head/adapter parameters
    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.structural_adapter.named_parameters()) + list(
        model.head.named_parameters()
    )

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in backbone_params if p.requires_grad],
            "lr": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in head_params if p.requires_grad],
            "lr": Config.HEAD_LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # --- Scheduler ---
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # --- Criterion ---
    # BCEWithLogitsLoss is suitable for both hard (0/1) and soft (float) labels
    criterion = nn.BCEWithLogitsLoss()

    # --- AWP Setup ---
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
    )

    best_score = -1.0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        avg_train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch, awp
        )

        # Validation
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        # Calculate AUC
        # Retrieve ground truth labels from the dataset wrapped by the loader
        valid_labels = valid_loader.dataset.labels
        if isinstance(valid_labels, torch.Tensor):
            valid_labels = valid_labels.cpu().numpy()

        score = get_score(valid_labels, preds.flatten())

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.8f}  avg_val_loss: {avg_val_loss:.8f}  AUC: {score:.8f}"
        )

        # Save Best Model
        if score > best_score:
            best_score = score
            print(f"Epoch {epoch+1} - Save Best Score: {best_score:.8f}")
            save_checkpoint(model, save_path)

    # Cleanup to free GPU memory
    del model, optimizer, scheduler, awp
    torch.cuda.empty_cache()
    gc.collect()

    return best_score
