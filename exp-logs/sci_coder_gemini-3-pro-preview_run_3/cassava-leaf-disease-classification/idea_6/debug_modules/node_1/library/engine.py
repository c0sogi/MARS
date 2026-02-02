import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

from library.config import Config
from library.utils import seed_everything, calculate_accuracy
from library.model import DualStreamModel


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, mixup_fn=None
):
    """
    Trains the model for one epoch using Gradient Accumulation and MixUp.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Handle empty loader edge case
    if len(loader) == 0:
        return 0.0, 0.0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup/CutMix
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

        # Recover unscaled loss for reporting
        running_loss += loss.item() * Config.ACCUMULATION_STEPS

        # Calculate Accuracy
        preds = outputs.argmax(dim=1)
        if mixup_fn is not None:
            targets = labels.argmax(dim=1)
        else:
            targets = labels

        correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if len(loader) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return running_loss / len(loader), correct / total if total > 0 else 0.0


def predict_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    """
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch depending on whether it includes image_id
            if len(batch) == 3:
                images, _, ids = batch
            else:
                images, _ = batch
                ids = []

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

    return image_ids, predictions


def train_model(
    train_loader,
    val_loader,
    device,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Orchestrates the training process including setup, loop, and early stopping.
    """
    print(f"Initializing training on {device}...")

    # Initialize Model
    model = DualStreamModel(pretrained=True)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
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

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Checkpoint & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! ({best_acc})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_acc


def generate_submission(
    test_loader,
    device,
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Generates submission file using the best model and TTA.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = DualStreamModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    print("Starting inference with TTA...")
    image_ids, preds = predict_tta(model, test_loader, device)

    df_sub = pd.DataFrame({"image_id": image_ids, "label": preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
