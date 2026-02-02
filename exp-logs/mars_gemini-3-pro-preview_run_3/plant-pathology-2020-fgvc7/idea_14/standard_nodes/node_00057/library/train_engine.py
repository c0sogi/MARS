import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import gc

from library.config import Config
from library.utils import seed_everything, get_class_weights, ModelEMA
from library.dataset import load_data, get_transforms, AppleDataset
from library.models import AppleNet
from library.losses import HybridLoss, DistillationLoss


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    loss_fn,
    teacher_model=None,
    ema=None,
    scheduler=None,
):
    """
    Runs one epoch of training.
    Handles both standard training (Teacher) and distillation (Student).
    """
    model.train()
    if teacher_model:
        teacher_model.eval()

    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            outputs = model(images)

            if teacher_model:
                # Distillation Mode
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                loss = loss_fn(outputs, teacher_outputs, targets)
            else:
                # Standard Mode
                loss = loss_fn(outputs, targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Computes Loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # Compute Loss (using HybridLoss logic inside)
            # Note: If loss_fn is DistillationLoss, it has a .hybrid_loss attribute we could use,
            # but usually we validate student on ground truth task loss.
            # Here we assume loss_fn passed to validate is always HybridLoss (Task Loss).
            loss = loss_fn(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions for ROC AUC
            # Main head logits -> Softmax
            probs = torch.softmax(outputs["main"], dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute ROC AUC
    # targets are one-hot/soft labels. sklearn handles this for multilabel/multiclass
    try:
        roc_auc = roc_auc_score(
            all_targets, all_preds, average="macro", multi_class="ovr"
        )
    except ValueError:
        # Handle edge cases where a class might not be present in the batch
        roc_auc = 0.5

    return epoch_loss, roc_auc


def train_teacher(train_df, val_df, class_weights):
    """
    Stage 1: Train the EfficientNetV2-M Teacher model.
    """
    print("\n" + "=" * 40)
    print("STARTING STAGE 1: TEACHER TRAINING")
    print("=" * 40)

    # Data Setup
    train_dataset = AppleDataset(
        train_df,
        transforms=get_transforms("train", Config.TEACHER_IMG_SIZE),
        mode="train",
    )
    val_dataset = AppleDataset(
        val_df,
        transforms=get_transforms("valid", Config.TEACHER_IMG_SIZE),
        mode="valid",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = AppleNet(Config.TEACHER_BACKBONE, pretrained=True)
    model.to(Config.DEVICE)

    # EMA Setup
    ema = ModelEMA(model)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )
    scaler = GradScaler()
    loss_fn = HybridLoss(weight=class_weights)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            Config.DEVICE,
            scaler,
            loss_fn,
            ema=ema,
            scheduler=scheduler,
        )

        # Validate using EMA model for stability
        val_loss, val_auc = validate(ema.module, val_loader, Config.DEVICE, loss_fn)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ema.module.state_dict(), Config.TEACHER_CHECKPOINT)
            print(f"  -> New Best Teacher Saved! AUC: {best_auc:.15f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Cleanup
    del model, ema, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    print("Teacher training complete.")


def train_student(train_df, val_df, class_weights):
    """
    Stage 2: Train the MaxViT-Small Student model via Distillation.
    """
    print("\n" + "=" * 40)
    print("STARTING STAGE 2: STUDENT DISTILLATION")
    print("=" * 40)

    # Data Setup (Student Resolution)
    train_dataset = AppleDataset(
        train_df,
        transforms=get_transforms("train", Config.STUDENT_IMG_SIZE),
        mode="train",
    )
    val_dataset = AppleDataset(
        val_df,
        transforms=get_transforms("valid", Config.STUDENT_IMG_SIZE),
        mode="valid",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Teacher
    print("Loading frozen Teacher model...")
    teacher_model = AppleNet(Config.TEACHER_BACKBONE, pretrained=False)
    teacher_model.load_state_dict(
        torch.load(Config.TEACHER_CHECKPOINT, map_location=Config.DEVICE)
    )
    teacher_model.to(Config.DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Setup Student
    student_model = AppleNet(Config.STUDENT_BACKBONE, pretrained=True)
    student_model.to(Config.DEVICE)

    # EMA Setup
    ema = ModelEMA(student_model)

    # Optimization
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )
    scaler = GradScaler()

    # Losses
    distill_loss_fn = DistillationLoss(
        weight=class_weights, alpha=Config.DISTILLATION_ALPHA, T=Config.TEMPERATURE
    )
    val_loss_fn = HybridLoss(weight=class_weights)  # Validate on pure task loss

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train with Distillation
        train_loss = train_one_epoch(
            student_model,
            train_loader,
            optimizer,
            Config.DEVICE,
            scaler,
            loss_fn=distill_loss_fn,
            teacher_model=teacher_model,
            ema=ema,
            scheduler=scheduler,
        )

        # Validate Student (using EMA)
        val_loss, val_auc = validate(ema.module, val_loader, Config.DEVICE, val_loss_fn)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ema.module.state_dict(), Config.STUDENT_CHECKPOINT)
            print(f"  -> New Best Student Saved! AUC: {best_auc:.15f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Cleanup
    del (
        teacher_model,
        student_model,
        ema,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
    )
    gc.collect()
    torch.cuda.empty_cache()
    print("Student training complete.")


def run_training():
    """
    Orchestrates the full training pipeline.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    print("Loading Metadata...")
    train_df = load_data(Config.TRAIN_CSV, "train_df", load_cached_data=True)
    val_df = load_data(Config.VAL_CSV, "val_df", load_cached_data=True)

    # Calculate Class Weights
    class_weights = get_class_weights(train_df, load_cached_data=True)
    print(f"Class Weights: {class_weights}")

    # Debug Mode: Subset data
    if Config.DEBUG:
        print("DEBUG MODE: Using subset of data")
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Stage 1: Train Teacher
    train_teacher(train_df, val_df, class_weights)

    # Stage 2: Train Student
    train_student(train_df, val_df, class_weights)

    print("Training pipeline finished.")
