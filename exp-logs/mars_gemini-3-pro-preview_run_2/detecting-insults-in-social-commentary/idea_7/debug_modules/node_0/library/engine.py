import time
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import AverageMeter, get_logger, time_since
from library.awp import AWP

logger = get_logger()


def train_mlm(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Task-Adaptive Pre-Training (TAPT) using Masked Language Modeling.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Forward pass (HuggingFace models compute MLM loss internally when labels are provided)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(train_loader):
            logger.info(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Elapsed: {time_since(start, (step + 1) / len(train_loader))} "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def train_fn(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Standard supervised training loop.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()
    criterion = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask)
        # Flatten outputs and targets for BCE
        loss = criterion(outputs.view(-1), targets.view(-1))

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(train_loader):
            logger.info(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Elapsed: {time_since(start, (step + 1) / len(train_loader))} "
                f"Loss: {losses.val:.8f} ({losses.avg:.8f})"
            )

    return losses.avg


def train_fn_awp(model, train_loader, optimizer, scheduler, device, epoch, awp):
    """
    Supervised training loop with Adversarial Weight Perturbation (AWP).
    """
    model.train()
    losses = AverageMeter()
    start = time.time()
    criterion = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        batch_size = input_ids.size(0)

        # 1. Forward and Backward (Clean)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1), targets.view(-1))

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        # 2. AWP Attack and Backward (Adversarial)
        if epoch >= Config.awp_start_epoch:
            # Save weights and apply perturbation based on gradients
            awp.attack_step()

            # Forward pass with perturbed weights
            adv_outputs = model(input_ids, attention_mask)
            adv_loss = criterion(adv_outputs.view(-1), targets.view(-1))

            # Backward pass for adversarial loss
            # Note: We rely on gradient accumulation or specific optimizer behavior.
            # Standard AWP pattern: accumulate gradients from adv pass, then restore weights.
            adv_loss.backward()

            # Restore original weights
            awp.restore()

        # 3. Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(train_loader):
            logger.info(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Elapsed: {time_since(start, (step + 1) / len(train_loader))} "
                f"Loss: {losses.val:.8f} ({losses.avg:.8f})"
            )

    return losses.avg


def evaluate_fn(model, val_loader, device):
    """
    Evaluation loop calculating Loss and AUC.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets_list = []

    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), targets.view(-1))

            losses.update(loss.item(), batch_size)

            # Apply sigmoid for predictions
            preds.append(torch.sigmoid(outputs).view(-1).cpu().numpy())
            targets_list.append(targets.view(-1).cpu().numpy())

    predictions = np.concatenate(preds)
    targets_all = np.concatenate(targets_list)

    try:
        auc_score = roc_auc_score(targets_all, predictions)
    except ValueError:
        auc_score = 0.5
        logger.warning(
            "ROC AUC Score could not be calculated (likely single class in batch). Defaulting to 0.5."
        )

    logger.info(
        f"EVAL: Loss: {losses.avg:.8f} | AUC: {auc_score:.8f} | "
        f"Elapsed: {time_since(start, 1)}"
    )

    return losses.avg, auc_score


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            preds.append(torch.sigmoid(outputs).view(-1).cpu().numpy())

            if (step + 1) % Config.print_freq == 0:
                logger.info(f"Inference step {step+1}/{len(test_loader)}")

    predictions = np.concatenate(preds)
    logger.info(f"Inference complete. Elapsed: {time_since(start, 1)}")
    return predictions


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
    use_awp=False,
):
    """
    Orchestrates the training loop with Early Stopping.
    """
    best_auc = 0.0
    best_loss = float("inf")
    early_stopping_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    # Initialize AWP if requested
    awp = None
    if use_awp:
        logger.info("Initializing AWP...")
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Select training function
        if use_awp:
            train_loss = train_fn_awp(
                model, train_loader, optimizer, scheduler, device, epoch, awp
            )
        else:
            train_loss = train_fn(
                model, train_loader, optimizer, scheduler, device, epoch
            )

        # Evaluation
        val_loss, val_auc = evaluate_fn(model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.8f}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            logger.info(f"Validation AUC Improved ({best_auc:.8f} ---> {val_auc:.8f})")
            best_auc = val_auc
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model Saved to {save_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(
                f"No improvement in AUC. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        if early_stopping_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break

    # Load best model before returning
    model.load_state_dict(best_model_weights)
    return model, best_auc
