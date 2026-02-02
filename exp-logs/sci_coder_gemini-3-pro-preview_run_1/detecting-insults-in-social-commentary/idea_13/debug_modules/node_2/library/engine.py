import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger, get_score
from library.awp import AWP

logger = get_logger("engine")


def train_fn(train_loader, model, optimizer, scheduler, device, epoch, awp=None):
    """
    Performs one epoch of training.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(input_ids, attention_mask, svd_features)
            loss = criterion(y_preds, labels)
            loss = loss / Config.grad_accum_steps

        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % Config.grad_accum_steps == 0:
            # Unscale gradients before AWP or clipping
            scaler.unscale_(optimizer)

            # Adversarial Weight Perturbation (AWP)
            if awp is not None and epoch >= Config.awp_start_epoch:
                awp.attack()
                with torch.cuda.amp.autocast(enabled=True):
                    y_preds_adv = model(input_ids, attention_mask, svd_features)
                    loss_adv = criterion(y_preds_adv, labels)
                    loss_adv = loss_adv / Config.grad_accum_steps

                # Accumulate adversarial gradients
                scaler.scale(loss_adv).backward()
                awp.restore()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * Config.grad_accum_steps * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def eval_fn(data_loader, model, device):
    """
    Performs inference on the validation set.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)

            batch_size = input_ids.size(0)

            y_preds = model(input_ids, attention_mask, svd_features)
            loss = criterion(y_preds, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds.append(y_preds.sigmoid().detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / dataset_size
    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    return avg_loss, predictions, ground_truth


def train_fold(
    model, train_loader, val_loader, optimizer, scheduler, device, fold_idx, model_name
):
    """
    Orchestrates the training for a single fold, including Early Stopping and Model Saving.
    """
    best_auc = -np.inf
    patience = 2
    counter = 0

    # Sanitize model name for filename
    sanitized_name = model_name.split("/")[-1]
    save_path = os.path.join(Config.MODEL_DIR, f"{sanitized_name}_fold_{fold_idx}.bin")

    # Initialize AWP
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    for epoch in range(Config.epochs):
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, device, epoch, awp
        )
        val_loss, val_preds, val_labels = eval_fn(val_loader, model, device)
        val_auc = get_score(val_labels, val_preds)

        logger.info(
            f"Fold {fold_idx} Epoch {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss} Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered")
                break

    return best_auc


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            y_preds = model(input_ids, attention_mask, svd_features)
            preds.append(y_preds.sigmoid().detach().cpu().numpy())

    return np.concatenate(preds)
