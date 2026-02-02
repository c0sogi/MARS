import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import MetricMonitor
from library.awp import AWP
from library.model import HybridModel


def get_optimizer_params(model, lr_backbone, lr_head, weight_decay):
    """
    Sets up differential learning rates for backbone and head.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "weight_decay": weight_decay,
            "lr": lr_head,
        },
    ]
    return optimizer_parameters


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    criterion,
    awp=None,
    scaler=None,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        svd_features = data["svd_features"].to(device)
        token_type_ids = data.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        targets = data["target"].to(device)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(input_ids, attention_mask, svd_features, token_type_ids)
            loss = criterion(logits.view(-1), targets)

        # Backward Pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adversarial Weight Perturbation (AWP)
        if awp is not None and epoch >= Config.awp_start_epoch:
            # AWP requires gradients to be populated
            awp.attack()

            # Forward pass with perturbed weights
            with torch.cuda.amp.autocast(enabled=True):
                logits_adv = model(
                    input_ids, attention_mask, svd_features, token_type_ids
                )
                loss_adv = criterion(logits_adv.view(-1), targets)

            # Backward pass for adversarial loss
            if scaler:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            # Restore original weights
            awp.restore()

        # Optimizer Step
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch {epoch} Train: {metric_monitor}")


def valid_one_epoch(epoch, model, val_loader, device, criterion):
    """
    Validates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    preds = []
    targets_list = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            svd_features = data["svd_features"].to(device)
            token_type_ids = data.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            targets = data["target"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(input_ids, attention_mask, svd_features, token_type_ids)
                loss = criterion(logits.view(-1), targets)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits.view(-1))
            preds.extend(probs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    auc = roc_auc_score(targets_list, preds)
    metric_monitor.update("AUC", auc)
    print(f"Epoch {epoch} Valid: {metric_monitor}")

    return np.array(preds), auc


def train_fold(fold, model_name, train_dataset, val_dataset):
    """
    Orchestrates the training process for a single fold.
    Returns the best validation predictions (OOF) for this fold.
    """
    print(f"=== Training Fold {fold} with model {model_name} ===")
    device = Config.device

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = HybridModel(model_name, pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer_grouped_parameters = get_optimizer_params(
        model, Config.lr_backbone, Config.lr_head, Config.weight_decay
    )
    optimizer = AdamW(optimizer_grouped_parameters)

    num_train_steps = int(len(train_dataset) / Config.train_batch_size * Config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Loss, Scaler, and AWP
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    awp = None
    if Config.use_awp:
        awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    # Tracking Best Performance
    best_auc = 0
    best_preds = None

    # Early Stopping
    patience = 2
    patience_counter = 0

    for epoch in range(1, Config.epochs + 1):
        train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            awp,
            scaler,
        )
        preds, auc = valid_one_epoch(epoch, model, val_loader, device, criterion)

        if auc > best_auc:
            print(f"AUC improved from {best_auc} to {auc}. Saving model...")
            best_auc = auc
            best_preds = preds

            # Save Model Checkpoint
            safe_model_name = model_name.replace("/", "_")
            save_path = os.path.join(
                Config.model_dir, f"{safe_model_name}_fold_{fold}.bin"
            )
            torch.save(model.state_dict(), save_path)

            patience_counter = 0
        else:
            patience_counter += 1
            print(f"AUC did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Cleanup to free GPU memory
    del model, optimizer, scheduler, scaler, awp
    torch.cuda.empty_cache()
    gc.collect()

    return best_preds


def predict(model, loader, device):
    """
    Runs inference on a dataset using a trained model.
    Useful for generating test set predictions.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            svd_features = data["svd_features"].to(device)
            token_type_ids = data.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(input_ids, attention_mask, svd_features, token_type_ids)

            probs = torch.sigmoid(logits.view(-1))
            preds.extend(probs.cpu().numpy())

    return np.array(preds)
