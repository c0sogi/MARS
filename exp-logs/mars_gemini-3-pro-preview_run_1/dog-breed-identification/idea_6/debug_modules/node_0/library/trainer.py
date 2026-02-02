import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold

from library.config import Config, seed_everything
from library.utils import MetricMonitor, save_checkpoint, get_checkpoint_path
from library.dataset import process_metadata, DogDataset, get_transforms
from library.model import DogClassifier
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item(), images.size(0))

    print(f"Epoch {epoch} [Train] {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def valid_one_epoch(model, loader, criterion, device, epoch):
    """
    Validates the model for one epoch.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()

            metric_monitor.update("Loss", loss.item(), images.size(0))
            metric_monitor.update("Accuracy", accuracy.item(), images.size(0))

    print(f"Epoch {epoch} [Valid] {metric_monitor}")
    return metric_monitor.get_avg("Loss"), metric_monitor.get_avg("Accuracy")


def predict(model, loader, device, use_tta=False):
    """
    Runs inference on the loader. Supports Test Time Augmentation (Horizontal Flip).
    """
    model.eval()
    probs_list = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            # Forward pass original
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            if use_tta:
                # Forward pass flipped
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.softmax(logits_flip, dim=1)

                # Average probabilities
                probs = 0.5 * (probs + probs_flip)

            probs_list.append(probs.cpu())

    return torch.cat(probs_list, dim=0)


def run_fold(fold_idx, train_df, val_df, classes, device):
    """
    Executes the 3-Phase training strategy for a single fold.
    """
    print(f"\n=== Starting Fold {fold_idx} ===")

    # 1. Setup Data
    # Note: process_metadata ensures 'label_idx' exists, so we don't need label_map for DogDataset here
    train_dataset = DogDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_dataset = DogDataset(val_df, transform=get_transforms("val"), mode="val")

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

    # 2. Setup Model & Loss
    model = DogClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    current_epoch = 1

    # ---------------------------------------------------------
    # Phase 1: Head Adaptation (Frozen Backbone)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Head Adaptation ---")
    model.set_backbone_trainable(False)

    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=Config.head_adapt_lr,
        weight_decay=Config.weight_decay,
    )

    for epoch in range(Config.head_adapt_epochs):
        train_one_epoch(
            model, train_loader, optimizer, criterion, device, current_epoch
        )
        valid_one_epoch(model, val_loader, criterion, device, current_epoch)
        current_epoch += 1

    # ---------------------------------------------------------
    # Phase 2: Fine-Tuning (Unfrozen Backbone, Discriminative LRs)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Fine-Tuning ---")
    model.set_backbone_trainable(True)

    # Discriminative Learning Rates
    backbone_params = [p for n, p in model.backbone.named_parameters()]
    head_params = [p for n, p in model.head.named_parameters()]

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.finetune_lr_backbone},
            {"params": head_params, "lr": Config.finetune_lr_head},
        ],
        weight_decay=Config.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.finetune_epochs
    )

    for epoch in range(Config.finetune_epochs):
        train_one_epoch(
            model, train_loader, optimizer, criterion, device, current_epoch
        )
        valid_one_epoch(model, val_loader, criterion, device, current_epoch)
        scheduler.step()
        current_epoch += 1

    # ---------------------------------------------------------
    # Phase 3: Stochastic Weight Averaging (SWA)
    # ---------------------------------------------------------
    print("\n--- Phase 3: SWA ---")

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.swa_lr)

    for epoch in range(Config.swa_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, current_epoch
        )

        # Update SWA model and scheduler
        swa_model.update_parameters(model)
        swa_scheduler.step()

        # Validate using the base model (optional, just to track progress)
        valid_one_epoch(model, val_loader, criterion, device, current_epoch)
        current_epoch += 1

    # Finalize SWA Model: Update Batch Normalization statistics
    print("Updating SWA Batch Normalization statistics...")
    update_bn(train_loader, swa_model, device=device)

    # Validate final SWA model
    print("Validating Final SWA Model...")
    val_loss, val_acc = valid_one_epoch(
        swa_model, val_loader, criterion, device, "SWA_Final"
    )

    # Save SWA Model
    ckpt_name = f"convnext_base_fold_{fold_idx}.pth"
    save_checkpoint(swa_model.state_dict(), ckpt_name)
    print(f"Saved SWA model for fold {fold_idx} to {ckpt_name}")

    return ckpt_name


def run_training_and_submission():
    """
    Main orchestrator function.
    1. Loads metadata and prepares K-Fold splits.
    2. Runs training for each fold.
    3. Performs inference on Test set (Ensemble of SWA models).
    4. Generates submission file.
    """
    seed_everything(Config.seed)
    device = Config.device

    # 1. Load and Prepare Data
    print("Loading metadata...")
    train_meta, val_meta, test_meta, classes = process_metadata(load_cached_data=True)

    # Combine provided train and val for Stratified K-Fold
    all_df = pd.concat([train_meta, val_meta], ignore_index=True)

    if Config.debug:
        print("DEBUG MODE: Subsetting data...")
        all_df = all_df.sample(n=200, random_state=Config.seed).reset_index(drop=True)
        test_meta = test_meta.head(50)
        Config.head_adapt_epochs = 1
        Config.finetune_epochs = 1
        Config.swa_epochs = 1
        Config.n_folds = 2

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    all_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df["breed"])):
        all_df.loc[val_idx, "fold"] = fold

    # 2. Train Folds
    model_paths = []
    for fold in range(Config.n_folds):
        fold_train_df = all_df[all_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = all_df[all_df["fold"] == fold].reset_index(drop=True)

        ckpt_path = run_fold(fold, fold_train_df, fold_val_df, classes, device)
        model_paths.append(ckpt_path)

    # 3. Inference & Ensemble
    print("\n=== Starting Inference & Ensemble ===")
    test_dataset = DogDataset(test_meta, transform=get_transforms("test"), mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Accumulate probabilities
    avg_probs = torch.zeros(len(test_meta), len(classes))

    for i, ckpt_name in enumerate(model_paths):
        print(f"Predicting with model fold {i}...")

        # Load Model
        model = DogClassifier(num_classes=len(classes), pretrained=False)
        # SWA model is saved as a module wrapper or state dict.
        # AveragedModel saves state_dict. If loaded into standard model, keys might match
        # if AveragedModel.module was used, or might have 'module.' prefix.
        # AveragedModel usually saves keys as 'n_averaged', 'module.backbone...', etc.
        # We need to handle loading carefully.

        state_dict = torch.load(get_checkpoint_path(ckpt_name), map_location=device)

        # Fix keys if they come from AveragedModel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove 'module.'
            elif k == "n_averaged":
                continue
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        # Predict with TTA
        fold_probs = predict(model, test_loader, device, use_tta=Config.use_tta)
        avg_probs += fold_probs

    # Average over folds
    avg_probs /= len(model_paths)

    # 4. Generate Submission
    print("Generating submission file...")
    submission_df = pd.DataFrame(avg_probs.numpy(), columns=classes)
    submission_df.insert(0, "id", test_meta["id"])

    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(submission_df.head())
