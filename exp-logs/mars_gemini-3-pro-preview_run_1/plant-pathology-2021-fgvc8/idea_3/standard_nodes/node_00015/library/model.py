import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import AppleDataset, get_transforms
from library.utils import (
    AverageMeter,
    calculate_metric,
    EarlyStopping,
    get_logger,
    seed_everything,
    save_checkpoint,
)


class AppleDiseaseSwinModel(nn.Module):
    def __init__(self, model_name=None, num_classes=None, pretrained=True):
        super().__init__()
        self.model_name = model_name if model_name else Config.model_name
        self.num_classes = num_classes if num_classes else Config.num_classes

        # Create model
        # Using drop_path_rate for Stochastic Depth as per Idea
        # Cite debug_lesson_3: Distinguish Initialization Arguments Between CNNs and Transformers
        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": self.num_classes,
            "drop_path_rate": 0.1,
        }

        # Only pass img_size for Transformer-based models that require it for partitioning
        if "swin" in self.model_name or "vit" in self.model_name:
            model_kwargs["img_size"] = Config.img_size

        self.model = timm.create_model(self.model_name, **model_kwargs)

    def forward(self, x):
        return self.model(x)


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with Label Smoothing.
    Smooths multi-label targets towards 0.5.
    """

    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.smoothing > 0:
            # Smooth targets: 1 -> 1 - eps + 0.5*eps, 0 -> 0.5*eps
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    score = calculate_metric(all_logits, all_labels, threshold=Config.threshold)
    return losses.avg, score


def train_model(debug=False):
    # Setup
    seed_everything(Config.seed)
    logger = get_logger(os.path.join(Config.working_dir, "train_log.txt"))
    device = Config.device

    logger.info(f"Starting training with model: {Config.model_name}")
    logger.info(f"Image Size: {Config.img_size}")

    # Data Loading
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    if debug:
        logger.info(f"Debug mode: using {Config.debug_sample_size} samples.")
        df_train = df_train.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)

    train_dataset = AppleDataset(
        df_train, mode="train", transform=get_transforms("train", Config.img_size)
    )
    val_dataset = AppleDataset(
        df_val, mode="val", transform=get_transforms("val", Config.img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model, Optimizer, Scheduler, Loss
    model = AppleDiseaseSwinModel(pretrained=True)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=Config.label_smoothing)
    scaler = GradScaler()

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=5, mode="max", save_path=Config.model_save_path, verbose=True
    )

    # Training Loop
    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F1: {val_score:.16f} - "
            f"Time: {elapsed:.0f}s"
        )

        early_stopping(val_score, model, optimizer, scheduler, epoch)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info("Training complete.")


def predict_and_submit():
    device = Config.device

    # Load Metadata
    df_test = pd.read_csv(Config.test_metadata_path)

    # Dataset & Loader
    test_dataset = AppleDataset(
        df_test, mode="test", transform=get_transforms("test", Config.img_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load Model
    model = AppleDiseaseSwinModel(pretrained=False)
    checkpoint_path = Config.model_save_path
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Cannot generate submission.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            with autocast():
                logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Process predictions
    pred_labels = []
    class_labels = Config.class_labels

    for probs in all_probs:
        # Get indices where prob > threshold
        indices = np.where(probs > Config.threshold)[0]

        if len(indices) == 0:
            # If no class exceeds threshold, pick the max probability class
            indices = [np.argmax(probs)]

        labels = [class_labels[i] for i in indices]
        pred_labels.append(" ".join(labels))

    # Create submission DataFrame
    df_sub = pd.DataFrame({"image": df_test["image"], "labels": pred_labels})

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
