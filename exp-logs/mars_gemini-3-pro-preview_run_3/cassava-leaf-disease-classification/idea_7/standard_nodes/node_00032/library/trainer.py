import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from library.config import CFG
from library.utils import AverageMeter, accuracy, save_checkpoint, get_logger
from library.models import CassavaClassifier


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, device):
    """
    Trains the model for one epoch using AMP and MixUp/CutMix.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # MixUp / CutMix Logic
        # We apply mixing with probability CFG.mixup_prob
        if np.random.rand() < CFG.mixup_prob:
            # 50% chance for MixUp, 50% chance for CutMix
            if np.random.rand() < 0.5:
                # MixUp
                lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
                rand_index = torch.randperm(batch_size).to(device)
                target_a = labels
                target_b = labels[rand_index]

                # Mix images
                images = lam * images + (1 - lam) * images[rand_index]

                with autocast():
                    output = model(images)
                    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(
                        output, target_b
                    )
            else:
                # CutMix
                lam = np.random.beta(CFG.cutmix_alpha, CFG.cutmix_alpha)
                rand_index = torch.randperm(batch_size).to(device)
                target_a = labels
                target_b = labels[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                # Adjust lambda to match exact pixel ratio
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (images.size(2) * images.size(3))
                )

                # Apply CutMix
                images[:, :, bbx1:bbx2, bby1:bby2] = images[
                    rand_index, :, bbx1:bbx2, bby1:bby2
                ]

                with autocast():
                    output = model(images)
                    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(
                        output, target_b
                    )
        else:
            # No mixing
            with autocast():
                output = model(images)
                loss = criterion(output, labels)

        # Optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def validate_one_epoch(val_loader, model, criterion, device):
    """
    Validates the model on the validation set.
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            output = model(images)
            loss = criterion(output, labels)

            acc1 = accuracy(output, labels, topk=(1,))[0]

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)

    return top1.avg, losses.avg


def fit(model_name, output_prefix, train_loader, val_loader):
    """
    Main training loop for a single model.

    Args:
        model_name (str): Name of the timm model to instantiate.
        output_prefix (str): Prefix for saved files (e.g., 'model_a').
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.

    Returns:
        float: Best validation accuracy achieved.
    """
    device = CFG.device

    # Setup Logging
    log_file = os.path.join(CFG.output_dir, f"{output_prefix}_train.log")
    logger = get_logger(log_file)

    logger.info(f"Initializing model: {model_name}")
    model = CassavaClassifier(model_name, CFG.num_classes, pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr
    )

    # Loss & Scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing).to(device)
    scaler = GradScaler()

    best_acc = 0.0
    patience_counter = 0
    best_model_path = f"{output_prefix}_best.pth"

    logger.info(
        f"Starting training for {CFG.epochs} epochs with patience {CFG.patience}"
    )

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, scaler, device
        )

        # Validate
        val_acc, val_loss = validate_one_epoch(val_loader, model, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Log metrics with full precision
        logger.info(
            f"Epoch {epoch+1}/{CFG.epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val Acc: {val_acc:.8f}"
        )

        # Save Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            logger.info(f"New best accuracy: {best_acc:.8f}. Saving model...")
        else:
            patience_counter += 1

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            CFG.output_dir, f"{output_prefix}_checkpoint.pth"
        )
        save_checkpoint(state, is_best, checkpoint_path, best_filepath=best_model_path)

        # Early Stopping
        if patience_counter >= CFG.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"Training finished for {output_prefix}. Best Val Acc: {best_acc:.8f}")
    return best_acc


def inference(model, loader, device):
    """
    Runs inference on a loader using the given model with Test-Time Augmentation (TTA).

    TTA Strategy:
    - Original
    - Horizontal Flip (if tta_steps >= 2)
    - Vertical Flip (if tta_steps >= 3)

    Returns:
        np.ndarray: Softmax probabilities of shape (N, num_classes)
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_probs = []

            # 1. Original View
            out = model(images)
            batch_probs.append(torch.softmax(out, dim=1))

            # 2. Horizontal Flip
            if CFG.tta_steps >= 2:
                out_h = model(torch.flip(images, dims=[3]))
                batch_probs.append(torch.softmax(out_h, dim=1))

            # 3. Vertical Flip
            if CFG.tta_steps >= 3:
                out_v = model(torch.flip(images, dims=[2]))
                batch_probs.append(torch.softmax(out_v, dim=1))

            # Average predictions across TTA views
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            probs.append(avg_probs.cpu().numpy())

    return np.concatenate(probs)


def generate_submission(model_a_path, model_b_path, test_loader):
    """
    Loads two trained models, runs inference with TTA, ensembles predictions via averaging,
    and saves the submission file.

    Args:
        model_a_path (str): Path to the best checkpoint for Model A (ViT).
        model_b_path (str): Path to the best checkpoint for Model B (EfficientNet).
        test_loader (DataLoader): DataLoader for the test set.
    """
    device = CFG.device
    logger = get_logger(os.path.join(CFG.output_dir, "submission.log"))

    # 1. Inference Model A
    logger.info(f"Loading Model A from {model_a_path}")
    model_a = CassavaClassifier(CFG.model_a_name, CFG.num_classes, pretrained=False)
    state_a = torch.load(model_a_path, map_location=device)
    model_a.load_state_dict(state_a["state_dict"])
    model_a.to(device)

    logger.info("Running inference for Model A...")
    probs_a = inference(model_a, test_loader, device)

    # Free memory
    del model_a, state_a
    torch.cuda.empty_cache()

    # 2. Inference Model B
    logger.info(f"Loading Model B from {model_b_path}")
    model_b = CassavaClassifier(CFG.model_b_name, CFG.num_classes, pretrained=False)
    state_b = torch.load(model_b_path, map_location=device)
    model_b.load_state_dict(state_b["state_dict"])
    model_b.to(device)

    logger.info("Running inference for Model B...")
    probs_b = inference(model_b, test_loader, device)

    # Free memory
    del model_b, state_b
    torch.cuda.empty_cache()

    # 3. Ensemble (Average Probabilities)
    logger.info("Ensembling predictions...")
    final_probs = (probs_a + probs_b) / 2.0
    final_preds = np.argmax(final_probs, axis=1)

    # 4. Save Submission
    test_df = pd.read_csv(CFG.test_csv)
    test_df["label"] = final_preds

    # Ensure only required columns
    submission_df = test_df[["image_id", "label"]]
    submission_df.to_csv(CFG.submission_file, index=False)

    logger.info(f"Submission saved to {CFG.submission_file}")
