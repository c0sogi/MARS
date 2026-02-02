import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torch_scatter import scatter_max
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import CdiscountDataset, collate_fn, get_category_mapping
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint,
    get_transforms,
)


class MultiViewResNet(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(MultiViewResNet, self).__init__()
        # Load ResNet18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace the FC layer with Identity to get the 512-dim feature vector
        # ResNet-18 structure: ... -> avgpool -> fc
        # We keep avgpool (Global Average Pooling on spatial dims) and remove fc.
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, batch_indices):
        """
        Args:
            x: (Total_Images, 3, H, W) - Flattened batch of all images
            batch_indices: (Total_Images,) - Tensor mapping each image to a product index in the batch
        """
        # Extract features for every image
        # Output: (Total_Images, 512)
        features = self.backbone(x)

        # Aggregate features using Global Max Pooling per product
        # batch_indices contains values from 0 to Batch_Size-1
        if batch_indices.numel() > 0:
            batch_size = batch_indices.max().item() + 1
        else:
            batch_size = 0

        # scatter_max returns a tuple (values, indices), we only need values
        # aggr_features: (Batch_Size, 512)
        aggr_features, _ = scatter_max(
            features, batch_indices, dim=0, dim_size=batch_size
        )

        # Classify
        logits = self.classifier(aggr_features)

        return logits


def train_model():
    # Setup device
    device = torch.device(Config.DEVICE)

    # Datasets
    train_dataset = CdiscountDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = CdiscountDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,
        transform=get_transforms("val"),
        mode="val",
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    # Model
    model = MultiViewResNet(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        # Training Phase
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for batch_idx, (images, indices, targets, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images, indices)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = calculate_accuracy(outputs, targets)
            train_loss.update(loss.item(), targets.size(0))
            train_acc.update(acc, targets.size(0))

        # Validation Phase
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for images, indices, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                indices = indices.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(images, indices)
                    loss = criterion(outputs, targets)

                acc = calculate_accuracy(outputs, targets)
                val_loss.update(loss.item(), targets.size(0))
                val_acc.update(acc, targets.size(0))

        # Metrics
        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss.avg:.6f} Train Acc: {train_acc.avg:.4f}% "
            f"Val Loss: {val_loss.avg:.6f} Val Acc: {val_acc.avg:.4f}%"
        )

        # Checkpointing
        is_best = val_acc.avg > best_acc
        if is_best:
            best_acc = val_acc.avg
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            },
            is_best,
        )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


def generate_submission():
    print("Generating submission...")
    device = torch.device(Config.DEVICE)

    # Load Model
    model = MultiViewResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    epoch, best_acc = load_checkpoint(Config.MODEL_CHECKPOINT, model)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {epoch} with Best Val Acc: {best_acc:.4f}%")

    # Test Dataset
    test_dataset = CdiscountDataset(
        metadata_path=Config.TEST_METADATA,
        bson_path=Config.TEST_BSON,
        transform=get_transforms("test"),
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    # Category Mapping
    _, idx_to_cat = get_category_mapping()

    predictions = []
    sample_ids = []

    with torch.no_grad():
        for images, indices, _, s_ids in test_loader:
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images, indices)

            # Get predictions
            _, preds = outputs.max(1)

            predictions.extend(preds.cpu().numpy())
            sample_ids.extend(s_ids.numpy())

    # Map indices back to category_ids
    category_ids = [idx_to_cat[p] for p in predictions]

    # Create DataFrame
    df_sub = pd.DataFrame({"_id": sample_ids, "category_id": category_ids})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    train_model()
    generate_submission()
