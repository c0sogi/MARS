import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from timm.data import Mixup

from library.utils import AverageMeter, accuracy, SoftTargetCrossEntropy
from library.data import get_dataloaders, get_test_dataloader
from library.model import get_model


def train_one_epoch(
    epoch, model, loader, optimizer, loss_fn, device, cfg, mixup_fn=None
):
    """
    Executes one training epoch with MixUp/CutMix augmentation.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply MixUp / CutMix if enabled
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # Compute accuracy (top-1)
            # accuracy() returns a list of accuracies for each topk, we take the first one
            acc1 = accuracy(outputs, targets, topk=(1,))[0]

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc1.item(), images.size(0))

    return loss_meter.avg, acc_meter.avg


def train_fold(fold_id, cfg):
    """
    Trains a model for a specific fold with Early Stopping and Checkpointing.
    """
    print(f"\n[Fold {fold_id}] Starting training...")

    device = cfg.device

    # DataLoaders
    train_loader, val_loader = get_dataloaders(fold_id, cfg)

    # Model
    model = get_model(cfg, pretrained=True)
    model.to(device)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    # Loss Functions
    # SoftTargetCrossEntropy for training (due to MixUp)
    # CrossEntropyLoss for validation (clean labels)
    train_loss_fn = SoftTargetCrossEntropy()
    val_loss_fn = nn.CrossEntropyLoss()

    # MixUp Function
    mixup_fn = None
    if cfg.mixup_prob > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=cfg.num_classes,
        )

    # Training Loop Variables
    best_acc = -1.0
    patience = 4  # Early stopping patience
    patience_counter = 0
    best_model_path = os.path.join(cfg.working_dir, f"fold_{fold_id}_best.pth")

    for epoch in range(cfg.epochs):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, train_loss_fn, device, cfg, mixup_fn
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, val_loss_fn, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Acc: {val_acc}"
        )

        # Checkpointing & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved for Fold {fold_id} with Accuracy: {best_acc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"[Fold {fold_id}] Finished. Best Validation Accuracy: {best_acc}")

    # Clean up to save memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()


def train_all_folds(cfg):
    """
    Orchestrates training for all folds.
    """
    for fold in range(cfg.n_folds):
        train_fold(fold, cfg)


def generate_submission(cfg):
    """
    Generates submission using 5-Fold Ensemble and TTA.
    """
    print("\nStarting Inference and Submission Generation...")

    device = cfg.device
    test_loader = get_test_dataloader(cfg)

    # Placeholder for ensemble predictions (num_samples, num_classes)
    num_test_samples = len(test_loader.dataset)
    ensemble_probs = torch.zeros((num_test_samples, cfg.num_classes), device=device)

    # Iterate over each fold model
    for fold in range(cfg.n_folds):
        model_path = os.path.join(cfg.working_dir, f"fold_{fold}_best.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for Fold {fold}...")
        model = get_model(cfg, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for images in test_loader:
                # CassavaDataset returns only images when output_label=False
                images = images.to(device)

                # Forward Pass (Original)
                logits = model(images)

                # TTA: Horizontal Flip
                if cfg.tta:
                    # flip last dimension (width)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    # Average logits as per requirement
                    logits = (logits + logits_flipped) / 2.0

                # Convert to probabilities
                probs = F.softmax(logits, dim=1)
                fold_probs.append(probs)

        # Concatenate batch results
        fold_probs = torch.cat(fold_probs, dim=0)

        # Add to ensemble
        ensemble_probs += fold_probs

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Average over folds
    ensemble_probs /= cfg.n_folds

    # Get final predictions
    final_preds = torch.argmax(ensemble_probs, dim=1).cpu().numpy()

    # Create Submission DataFrame
    test_df = pd.read_csv(cfg.test_metadata_path)
    submission_df = pd.DataFrame(
        {"image_id": test_df["image_id"], "label": final_preds}
    )

    # Save
    save_path = os.path.join(cfg.submission_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())
