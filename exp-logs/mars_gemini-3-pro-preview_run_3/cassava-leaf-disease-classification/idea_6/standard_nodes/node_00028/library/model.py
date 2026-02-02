import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataset


class CassavaModel(nn.Module):
    """
    Single-Stream Vision Transformer.
    Cite solution_lesson_node_00025: Prefer ensembling independent models over jointly training multi-stream architectures.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            img_size=Config.IMG_SIZE,
            drop_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        return self.model(x)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, mixup_fn
):
    """
    Handles training for one epoch with Gradient Accumulation and Mixup.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup/CutMix if enabled
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Scale loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step (Accumulated)
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step scheduler every update
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * Config.ACCUMULATION_STEPS

        # Calculate Accuracy
        # If Mixup is used, we compare argmax of preds and targets
        preds = outputs.argmax(dim=1)
        if mixup_fn is not None:
            targets = labels.argmax(dim=1)
        else:
            targets = labels

        correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Handles validation loop.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / len(loader), correct / total


def train_model():
    """
    Main training routine.
    """
    # Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Initializing training on {device}...")

    # Datasets & Loaders
    train_dataset = get_dataset("train", debug=Config.DEBUG)
    val_dataset = get_dataset("val", debug=Config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = DualStreamModel(pretrained=True)
    model = model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    # Total steps = (epochs * steps_per_epoch) / accumulation
    steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    total_steps = Config.EPOCHS * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=Config.MIN_LR
    )

    # Loss Functions & Mixup
    mixup_fn = None
    if Config.MIX_PROB > 0:
        mixup_fn = Mixup(
            mixup_alpha=Config.MIXUP_ALPHA,
            cutmix_alpha=Config.CUTMIX_ALPHA,
            prob=Config.MIX_PROB,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=Config.LABEL_SMOOTHING,
            num_classes=Config.NUM_CLASSES,
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = LabelSmoothingCrossEntropy(smoothing=Config.LABEL_SMOOTHING)

    val_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training Loop
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            train_criterion,
            device,
            scaler,
            mixup_fn,
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! ({best_acc:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


def predict():
    """
    Inference routine with Test-Time Augmentation (TTA).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_dataset = get_dataset("test", debug=Config.DEBUG)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model = DualStreamModel(pretrained=False)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    image_ids = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for images, _, ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)

            # 1. Original
            out = model(images)
            probs = torch.softmax(out, dim=1)

            if Config.USE_TTA:
                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                out_h = model(images_h)
                probs += torch.softmax(out_h, dim=1)

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                out_v = model(images_v)
                probs += torch.softmax(out_v, dim=1)

                # Average
                probs /= 3.0

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            predictions.extend(preds)
            image_ids.extend(ids)

    # Save Submission
    df_sub = pd.DataFrame({"image_id": image_ids, "label": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
