import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, accuracy, seed_everything
from library.dataset import CachedFeatureDataset
from library.model import PDFCNet


def mixup_data(x, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam


def mixup_criterion(criterion, preds, targets_a, targets_b, lam):
    """
    Computes mixup loss for hierarchical outputs.
    preds: list of [logits1, logits2, logits3]
    targets_a: list of [l1, l2, l3] (original)
    targets_b: list of [l1, l2, l3] (shuffled)
    """
    total_loss = 0
    # Sum loss over all 3 levels
    for pred, ta, tb in zip(preds, targets_a, targets_b):
        total_loss += lam * criterion(pred, ta) + (1 - lam) * criterion(pred, tb)
    return total_loss


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Feature-Space MixUp.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    # We track L3 accuracy on the primary target for monitoring,
    # though strictly speaking accuracy on mixed labels is ambiguous.
    # We calculate it against the 'primary' label (targets_a) for rough guidance.
    top1 = AverageMeter("Acc@1", ":6.2f")

    start_time = time.time()

    for i, (features, l1, l2, l3) in enumerate(train_loader):
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        # Apply MixUp
        mixed_features, index, lam = mixup_data(features, Config.MIXUP_ALPHA, device)

        # Forward pass
        logits1, logits2, logits3 = model(mixed_features)

        # Prepare targets for mixup criterion
        targets_a = [l1, l2, l3]
        targets_b = [l1[index], l2[index], l3[index]]
        preds = [logits1, logits2, logits3]

        # Compute Loss
        loss = mixup_criterion(criterion, preds, targets_a, targets_b, lam)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), features.size(0))

        # Measure accuracy for L3 against the dominant label
        acc1 = accuracy(logits3, l3, topk=(1,))
        top1.update(acc1[0].item(), features.size(0))

    epoch_time = time.time() - start_time
    print(
        f"Epoch: [{epoch}] Train Loss: {losses.avg} Train L3 Acc: {top1.avg} Time: {epoch_time:.2f}s"
    )
    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1_l1 = AverageMeter("L1 Acc@1", ":6.2f")
    top1_l2 = AverageMeter("L2 Acc@1", ":6.2f")
    top1_l3 = AverageMeter("L3 Acc@1", ":6.2f")

    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            l3 = l3.to(device)

            # Forward pass
            logits1, logits2, logits3 = model(features)

            # Compute Loss (Sum of all levels)
            loss = (
                criterion(logits1, l1) + criterion(logits2, l2) + criterion(logits3, l3)
            )

            # Update metrics
            losses.update(loss.item(), features.size(0))

            acc1_l1 = accuracy(logits1, l1, topk=(1,))
            top1_l1.update(acc1_l1[0].item(), features.size(0))

            acc1_l2 = accuracy(logits2, l2, topk=(1,))
            top1_l2.update(acc1_l2[0].item(), features.size(0))

            acc1_l3 = accuracy(logits3, l3, topk=(1,))
            top1_l3.update(acc1_l3[0].item(), features.size(0))

    print(f"Validation Results - Loss: {losses.avg}")
    print(f"  L1 Acc: {top1_l1.avg}")
    print(f"  L2 Acc: {top1_l2.avg}")
    print(f"  L3 Acc: {top1_l3.avg}")

    return top1_l3.avg, losses.avg


def train_model():
    """
    Main function to execute the training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Loading datasets...")
    train_dataset = CachedFeatureDataset(
        feature_path=Config.TRAIN_FEATURES_PATH,
        label_path=Config.TRAIN_LABELS_PATH,
        is_test=False,
    )

    val_dataset = CachedFeatureDataset(
        feature_path=Config.VAL_FEATURES_PATH,
        label_path=Config.VAL_LABELS_PATH,
        is_test=False,
    )

    # Debugging option
    if Config.DEBUG:
        print(f"DEBUG: Subsetting datasets to {Config.DEBUG_SIZE} samples.")
        indices = list(range(min(len(train_dataset), Config.DEBUG_SIZE)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 2. Initialize Model
    model = PDFCNet().to(device)

    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_acc = -1.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_acc, val_loss = validate(val_loader, model, criterion, device)

        # Checkpointing
        if val_acc > best_acc:
            print(
                f"Validation Accuracy improved from {best_acc} to {val_acc}. Saving model..."
            )
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best L3 Accuracy: {best_acc}")
