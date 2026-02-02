import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, AverageMeter, get_score
from library.dataset import CassavaDataset, get_transforms


class CassavaSwinModel(nn.Module):
    """
    Swin Transformer model for Cassava Leaf Disease Classification.
    Uses a pre-trained Swin Large backbone and replaces the head for 5-class classification.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(CassavaSwinModel, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# Helper Functions for Mixup / Cutmix
# =============================================================================


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))

    return x, y, y[index], lam


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Training and Validation Functions
# =============================================================================


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    losses = AverageMeter()

    # Gradient accumulation setup
    accum_steps = Config.ACCUMULATION_STEPS
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # MixUp / CutMix Decision
        do_mixup = False
        do_cutmix = False
        if np.random.rand() < Config.MIXUP_PROB:
            if np.random.rand() < 0.5:
                do_mixup = True
            else:
                do_cutmix = True

        with autocast():
            if do_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, Config.MIXUP_ALPHA
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif do_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, labels, Config.CUTMIX_ALPHA
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss for gradient accumulation
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.update(loss.item() * accum_steps, batch_size)

    print(f"Train Epoch: {epoch} | Loss: {losses.avg:.6f}")
    return losses.avg


def valid_one_epoch(epoch, model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            acc = get_score(labels.cpu().numpy(), preds.cpu().numpy())

            losses.update(loss.item(), batch_size)
            scores.update(acc, batch_size)

    print(f"Valid Epoch: {epoch} | Loss: {losses.avg:.6f} | Acc: {scores.avg:.6f}")
    return losses.avg, scores.avg


# =============================================================================
# Main Execution Functions
# =============================================================================


def run_training():
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        df_train = df_train.sample(n=100, random_state=Config.SEED).reset_index(
            drop=True
        )
        df_val = df_val.sample(n=50, random_state=Config.SEED).reset_index(drop=True)

    # Datasets and Loaders
    train_dataset = CassavaDataset(
        df_train, transforms=get_transforms("train"), output_label=True
    )
    val_dataset = CassavaDataset(
        df_val, transforms=get_transforms("valid"), output_label=True
    )

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

    # Model Setup
    device = Config.DEVICE
    model = CassavaSwinModel()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eps=Config.EPS,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    scaler = GradScaler()

    # Training Loop
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = valid_one_epoch(epoch, model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Time: {elapsed:.2f}s")

        # Save Best Model
        if val_acc > best_acc:
            print(f"Validation Accuracy Improved ({best_acc:.6f} ---> {val_acc:.6f})")
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in validation accuracy. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Accuracy: {best_acc:.6f}")


def predict():
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Dataset
    test_dataset = CassavaDataset(
        df_test, transforms=get_transforms("test"), output_label=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    device = Config.DEVICE
    model = CassavaSwinModel(
        pretrained=False
    )  # Pretrained weights not needed when loading state_dict

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Error: Model file not found at {Config.MODEL_SAVE_PATH}. Cannot predict."
        )
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []

    # Inference with TTA
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass 1: Original
            out1 = model(images)

            # Forward pass 2: Horizontal Flip (TTA)
            if Config.USE_TTA:
                images_flip = torch.flip(images, dims=[3])  # [B, C, H, W], flip W
                out2 = model(images_flip)
                outputs = (out1 + out2) / 2.0
            else:
                outputs = out1

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Save Submission
    df_test["label"] = predictions
    # Keep only required columns
    submission = df_test[["image_id", "label"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
