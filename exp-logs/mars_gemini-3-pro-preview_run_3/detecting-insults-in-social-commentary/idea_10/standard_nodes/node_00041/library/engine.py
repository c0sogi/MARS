import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.awp import AWP
from library.utils import seed_everything


def train_teacher_fn(
    model, train_loader, val_loader, optimizer, scheduler, device, config, save_path
):
    """
    Training loop for Teacher models using standard BCE loss on labeled data.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0

    # Ensure model is in training mode
    model.train()

    for epoch in range(config.teacher_epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["labels"].to(device, dtype=torch.float)

            # Forward pass
            logits = model(input_ids=ids, attention_mask=mask)
            loss = criterion(logits, targets)

            # Normalize loss for gradient accumulation
            loss = loss / config.accumulation_steps
            loss.backward()

            train_loss += loss.item() * config.accumulation_steps

            if (step + 1) % config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

        # Validation
        val_auc, val_loss = validate_fn(model, val_loader, device, criterion)

        print(
            f"Teacher Epoch {epoch+1}/{config.teacher_epochs} | Train Loss: {train_loss/len(train_loader)} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    # Load best model state before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_auc


def train_student_awp_fn(
    model,
    labeled_loader,
    distillation_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    config,
    save_path,
):
    """
    Training loop for Student models using Hybrid Loss (BCE + Soft Targets) and Adversarial Weight Perturbation (AWP).
    """
    # Criterion for labeled data (Hard Labels)
    criterion_hard = nn.BCEWithLogitsLoss()
    # Criterion for distillation data (Soft Targets)
    # BCEWithLogitsLoss is appropriate here as soft_targets are probabilities
    criterion_soft = nn.BCEWithLogitsLoss()

    # Initialize AWP
    awp = AWP(model, config, optimizer, adv_param="weight")
    best_auc = 0.0

    for epoch in range(config.student_epochs):
        model.train()
        train_loss = 0.0

        # Create iterator for distillation data to cycle through it
        distill_iter = iter(distillation_loader)

        for step, labeled_batch in enumerate(labeled_loader):
            # Fetch distillation batch
            try:
                distill_batch = next(distill_iter)
            except StopIteration:
                distill_iter = iter(distillation_loader)
                distill_batch = next(distill_iter)

            # --- Forward Pass (Labeled Data) ---
            ids_l = labeled_batch["input_ids"].to(device, dtype=torch.long)
            mask_l = labeled_batch["attention_mask"].to(device, dtype=torch.long)
            targets_l = labeled_batch["labels"].to(device, dtype=torch.float)

            logits_l = model(input_ids=ids_l, attention_mask=mask_l)
            loss_l = criterion_hard(logits_l, targets_l)

            # --- Forward Pass (Distillation Data) ---
            ids_d = distill_batch["input_ids"].to(device, dtype=torch.long)
            mask_d = distill_batch["attention_mask"].to(device, dtype=torch.long)
            targets_d = distill_batch["soft_targets"].to(device, dtype=torch.float)

            logits_d = model(input_ids=ids_d, attention_mask=mask_d)
            loss_d = criterion_soft(logits_d, targets_d)

            # --- Hybrid Loss Calculation ---
            loss = loss_l + config.distillation_alpha * loss_d

            # Normalize loss
            loss = loss / config.accumulation_steps
            loss.backward()

            train_loss += loss.item() * config.accumulation_steps

            # --- Adversarial Weight Perturbation (AWP) ---
            if config.use_awp and epoch >= config.awp_start_epoch:
                # 1. Perturb weights to maximize loss
                awp.attack()

                # 2. Re-compute forward pass and loss with perturbed weights
                # We re-compute for both labeled and distillation parts to ensure robustness across the board
                logits_l_adv = model(input_ids=ids_l, attention_mask=mask_l)
                loss_l_adv = criterion_hard(logits_l_adv, targets_l)

                logits_d_adv = model(input_ids=ids_d, attention_mask=mask_d)
                loss_d_adv = criterion_soft(logits_d_adv, targets_d)

                loss_adv = loss_l_adv + config.distillation_alpha * loss_d_adv
                loss_adv = loss_adv / config.accumulation_steps

                # 3. Backward pass on adversarial loss
                loss_adv.backward()

                # 4. Restore original weights
                awp.restore()

            # --- Optimizer Step ---
            if (step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

        # Validation
        val_auc, val_loss = validate_fn(model, val_loader, device, criterion_hard)

        print(
            f"Student Epoch {epoch+1}/{config.student_epochs} | Train Loss: {train_loss/len(labeled_loader)} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    # Load best model state before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_auc


def validate_fn(model, val_loader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    final_targets = []
    final_outputs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["labels"].to(device, dtype=torch.float)

            logits = model(input_ids=ids, attention_mask=mask)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            final_targets.extend(targets.cpu().numpy().tolist())
            final_outputs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader)

    try:
        auc = roc_auc_score(final_targets, final_outputs)
    except ValueError:
        auc = 0.0

    return auc, avg_loss


def predict_fn(model, loader, device):
    """
    Generates probability predictions for the given data loader.
    """
    model.eval()
    final_outputs = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)

            logits = model(input_ids=ids, attention_mask=mask)
            probs = torch.sigmoid(logits)

            final_outputs.extend(probs.cpu().numpy().tolist())

    return np.array(final_outputs)
