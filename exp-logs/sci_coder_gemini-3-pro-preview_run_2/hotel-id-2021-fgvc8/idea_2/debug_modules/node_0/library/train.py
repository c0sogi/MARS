import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import HotelDataset, get_transforms, get_class_to_idx
from library.model import HotelRecognitionModel
from library.utils import AverageMeter, save_checkpoint, seed_everything


def train_fn(dataloader, model, criterion, optimizer, device, epoch, scheduler):
    """
    Training function for one epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # Get total batches for logging
    num_batches = len(dataloader)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # When labels are provided, model returns ArcFace logits
        logits = model(images, labels)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

        # Calculate accuracy (top-1)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        acc_meter.update(acc.item(), batch_size)

        if step % Config.print_freq == 0 or step == (num_batches - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{num_batches}] "
                f"Loss: {loss_meter.val} ({loss_meter.avg}) "
                f"Acc: {acc_meter.val} ({acc_meter.avg}) "
                f"LR: {optimizer.param_groups[0]['lr']}"
            )

    return loss_meter.avg, acc_meter.avg


def validate_fn(dataloader, model, criterion, device):
    """
    Validation function.
    Computes Loss and Accuracy on the validation set using the ArcFace head.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images, labels)

            # Compute loss
            loss = criterion(logits, labels)

            # Update metrics
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)

            # Calculate accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
            acc_meter.update(acc.item(), batch_size)

    return loss_meter.avg, acc_meter.avg


def run_training(debug=False, epochs=Config.epochs):
    """
    Main training execution function.

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
        epochs (int): Number of epochs to train.
    """
    seed_everything(Config.seed)

    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)

    if debug:
        print("DEBUG Mode: Subsampling data...")
        train_df = train_df.sample(n=500, random_state=Config.seed).reset_index(
            drop=True
        )
        val_df = val_df.sample(n=100, random_state=Config.seed).reset_index(drop=True)
        # Ensure we don't have issues with missing classes in debug mode by regenerating mapping later
        # But for model compatibility, we must stick to Config.num_classes.
        # In debug, some classes might not be present in the batch, which is fine.

    # 2. Prepare Class Mapping
    # We generate the mapping from the full training set (or the provided metadata)
    # to ensure consistency.
    print("Generating class mapping...")
    # Note: We use the original full train_df for mapping generation to ensure
    # indices match Config.num_classes logic if we were using the full set.
    # However, since we are loading from metadata that covers the distribution,
    # we can just use the loaded train_df (if not debug) or read it fresh.
    if debug:
        full_train_df = pd.read_csv(Config.train_metadata_path)
        class_to_idx = get_class_to_idx(full_train_df)
    else:
        class_to_idx = get_class_to_idx(train_df)

    print(f"Number of classes: {len(class_to_idx)}")

    # 3. Create Datasets and Dataloaders
    train_dataset = HotelDataset(
        df=train_df,
        transform=get_transforms(mode="train"),
        data_root=Config.input_dir,
        mode="train",
        class_to_idx=class_to_idx,
    )

    val_dataset = HotelDataset(
        df=val_df,
        transform=get_transforms(mode="val"),
        data_root=Config.input_dir,
        mode="val",
        class_to_idx=class_to_idx,
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
        drop_last=False,
    )

    # 4. Initialize Model
    print(f"Initializing model: {Config.backbone_name}")
    model = HotelRecognitionModel(
        n_classes=Config.num_classes,
        backbone_name=Config.backbone_name,
        pretrained=Config.pretrained,
        embedding_size=Config.embedding_size,
    )
    model.to(Config.device)

    # 5. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.min_lr)

    # 6. Loss Function
    criterion = nn.CrossEntropyLoss()

    # 7. Training Loop
    best_loss = float("inf")

    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_fn(
            train_loader, model, criterion, optimizer, Config.device, epoch, scheduler
        )

        # Validate
        val_loss, val_acc = validate_fn(val_loader, model, criterion, Config.device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1} Complete. Time: {elapsed}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Train Acc:  {train_acc}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Acc:    {val_acc}")

        # Save Checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            print(f"  New Best Validation Loss: {best_loss}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            Config.model_save_path,
        )

    print("Training complete.")
