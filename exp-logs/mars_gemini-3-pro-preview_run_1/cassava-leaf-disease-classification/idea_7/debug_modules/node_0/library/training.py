import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import AverageMeter, get_logger, save_checkpoint, accuracy
from library.loss import SoftTargetCrossEntropy
from library.data import CassavaDataset, get_transforms, MixupCollate
from library.modeling import CassavaClassifier, get_llrd_params


def get_dataloaders(train_df, val_df, img_size, batch_size):
    """
    Creates DataLoaders for training and validation with specific image size.
    Used to switch resolutions during Progressive Resizing.
    """
    # Training Data
    train_dataset = CassavaDataset(
        train_df, transform=get_transforms(img_size, mode="train"), mode="train"
    )

    # MixUp Collate
    mixup_fn = MixupCollate(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        num_classes=Config.NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=mixup_fn,
        worker_init_fn=lambda x: np.random.seed(Config.SEED + x),
    )

    # Validation Data
    val_dataset = CassavaDataset(
        val_df, transform=get_transforms(img_size, mode="valid"), mode="valid"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def train_one_epoch(epoch, model, loader, optimizer, criterion, scaler, device, logger):
    """
    Handles one epoch of training with AMP, Gradient Accumulation, and MixUp.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":.2f")

    # Zero gradients initially
    optimizer.zero_grad()

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed Precision Forward
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.ACCUM_STEPS

        # Backward
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (batch_idx + 1) % Config.ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        # Restore loss value for logging
        loss_val = loss.item() * Config.ACCUM_STEPS
        losses.update(loss_val, images.size(0))

        # Approximate accuracy for MixUp targets (using argmax)
        # targets is (B, C) due to MixUp
        with torch.no_grad():
            hard_targets = targets.argmax(dim=1)
            acc1 = accuracy(outputs, hard_targets, topk=(1,))[0]
            top1.update(acc1.item(), images.size(0))

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch} [Train] Loss: {losses.avg:.6f} Acc: {top1.avg:.6f} Time: {elapsed:.1f}s"
    )

    return losses.avg, top1.avg


def valid_one_epoch(epoch, model, loader, criterion, device, logger):
    """
    Handles validation. No MixUp, standard CrossEntropy behavior.
    """
    model.eval()

    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":.2f")

    start_time = time.time()

    # Validation criterion is usually standard CrossEntropy (hard labels)
    # But if we use SoftTargetCrossEntropy in training, we need to ensure compatibility.
    # The loader returns hard labels (LongTensor) for validation.
    # Standard nn.CrossEntropyLoss handles LongTensor targets.
    val_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = val_criterion(outputs, targets)

            acc1 = accuracy(outputs, targets, topk=(1,))[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch} [Valid] Loss: {losses.avg:.6f} Acc: {top1.avg:.6f} Time: {elapsed:.1f}s"
    )

    return losses.avg, top1.avg


class SWAContainer:
    """
    Helper to manage Stochastic Weight Averaging.
    """

    def __init__(self, model, device):
        self.swa_model = AveragedModel(model).to(device)
        self.device = device

    def update(self, model):
        self.swa_model.update_parameters(model)

    def finalize(self, train_loader):
        """
        Updates Batch Normalization statistics using the training data.
        """
        print("SWA: Updating BN statistics...")
        # We need a loader that returns just images or handles the format correctly
        # update_bn expects loader to yield image batches
        # Our loader yields (images, targets)
        # We wrap it to yield just images

        # Note: update_bn runs in forward mode, so we need to ensure the model is in train mode
        # but gradients are disabled. update_bn handles this internally usually.
        update_bn(train_loader, self.swa_model, device=self.device)

    def get_model(self):
        return self.swa_model.module


def run_training(model_arch, logger_name="train.log"):
    """
    Main pipeline for training a single model architecture.
    """
    logger = get_logger(os.path.join(Config.OUTPUT_DIR, logger_name))
    logger.info(f"Starting training for architecture: {model_arch}")

    device = Config.DEVICE

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # 2. Initialize Model
    model = CassavaClassifier(model_arch, pretrained=True)
    model.to(device)

    # 3. Optimizer & Scheduler
    # Use Layer-wise Learning Rate Decay
    param_groups = get_llrd_params(
        model,
        lr=Config.LR_MAX,
        weight_decay=Config.WEIGHT_DECAY,
        decay_factor=Config.LLRD_DECAY,
    )
    optimizer = optim.AdamW(param_groups)
    scaler = GradScaler()

    # Criterion
    criterion = SoftTargetCrossEntropy()

    # --- PHASE 1: Base Training (Low Res) ---
    logger.info(
        f"\n--- Phase 1: Base Training ({Config.IMG_SIZE_LOW}x{Config.IMG_SIZE_LOW}) ---"
    )

    train_loader, val_loader = get_dataloaders(
        train_df, val_df, Config.IMG_SIZE_LOW, Config.BATCH_SIZE
    )

    # Scheduler: Cosine Annealing for Base + Fine epochs
    total_epochs_main = Config.EPOCHS_WARMUP + Config.EPOCHS_BASE + Config.EPOCHS_FINE
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs_main, eta_min=1e-6
    )

    best_acc = 0.0

    # Warmup + Base
    for epoch in range(1, Config.EPOCHS_WARMUP + Config.EPOCHS_BASE + 1):
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scaler, device, logger
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, val_loader, criterion, device, logger
        )

        scheduler.step()

        # Save Best Base
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model.state_dict(),
                True,
                os.path.join(Config.OUTPUT_DIR, f"{model_arch}_base_best.pth"),
            )

    # --- PHASE 2: Fine-Tuning (High Res) ---
    logger.info(
        f"\n--- Phase 2: Fine-Tuning ({Config.IMG_SIZE_HIGH}x{Config.IMG_SIZE_HIGH}) ---"
    )

    # Re-initialize loaders with high resolution
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, Config.IMG_SIZE_HIGH, Config.BATCH_SIZE
    )

    best_acc_fine = 0.0

    for epoch in range(
        Config.EPOCHS_WARMUP + Config.EPOCHS_BASE + 1, total_epochs_main + 1
    ):
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scaler, device, logger
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, val_loader, criterion, device, logger
        )

        scheduler.step()

        if val_acc > best_acc_fine:
            best_acc_fine = val_acc
            save_checkpoint(
                model.state_dict(),
                True,
                os.path.join(Config.OUTPUT_DIR, f"{model_arch}_fine_best.pth"),
            )

    # --- PHASE 3: SWA (High Res) ---
    logger.info(f"\n--- Phase 3: SWA Training ---")

    swa_container = SWAContainer(model, device)

    # Switch to SWA LR
    # We re-initialize optimizer or just set param groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.SWA_LR

    start_swa_epoch = total_epochs_main + 1
    end_swa_epoch = start_swa_epoch + Config.EPOCHS_SWA

    for epoch in range(start_swa_epoch, end_swa_epoch):
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scaler, device, logger
        )

        # Update SWA Model
        swa_container.update(model)

        # Optional: Evaluate current model (not SWA yet)
        valid_one_epoch(epoch, model, val_loader, criterion, device, logger)

    # Finalize SWA
    logger.info("Finalizing SWA Model...")
    swa_container.finalize(train_loader)

    # Validate SWA Model
    swa_model = swa_container.get_model()
    val_loss, val_acc = valid_one_epoch(
        "SWA_Final", swa_model, val_loader, criterion, device, logger
    )

    logger.info(f"Final SWA Accuracy: {val_acc:.6f}")

    # Save SWA Model
    save_checkpoint(
        swa_model.state_dict(),
        False,
        os.path.join(Config.OUTPUT_DIR, f"{model_arch}_swa_final.pth"),
    )

    return val_acc
