import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import AverageMeter, get_class_weights, seed_everything
from library.dataset import get_loaders
from library.models import HierarchicalEfficientNet, HierarchicalSwin


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state.
    """

    def __init__(
        self, patience=Config.PATIENCE, verbose=False, delta=0, path="checkpoint.pth"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_fn(dataloader, model, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def valid_fn(dataloader, model, criterion, device):
    """
    Executes validation loop and calculates ROC AUC.
    """
    model.eval()
    loss_meter = AverageMeter()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            # Apply softmax to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate Macro ROC AUC (Column-wise mean)
    try:
        roc_auc = roc_auc_score(targets, preds, average="macro", multi_class="ovr")
    except ValueError:
        # Handle edge case where a batch/fold might miss a class (unlikely with stratification)
        roc_auc = 0.5

    return loss_meter.avg, roc_auc


def run_fold(fold, model_type):
    """
    Trains a specific model architecture for a specific fold.

    Args:
        fold (int): Fold index (0-4).
        model_type (str): 'effnet' or 'swin'.
    """
    seed_everything(Config.SEED)

    device = Config.DEVICE

    # Configure based on model type
    if model_type == "effnet":
        model = HierarchicalEfficientNet(pretrained=True)
        img_size = Config.IMG_SIZE_EFFNET
        save_name = f"effnet_fold_{fold}_best.pth"
    elif model_type == "swin":
        model = HierarchicalSwin(pretrained=True)
        img_size = Config.IMG_SIZE_SWIN
        save_name = f"swin_fold_{fold}_best.pth"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    # Get DataLoaders
    train_loader, val_loader = get_loaders(
        fold=fold, image_size=img_size, batch_size=Config.BATCH_SIZE
    )

    # Loss Function with Class Weights
    class_weights = get_class_weights(load_cached_data=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Early Stopping
    save_path = os.path.join(Config.WORK_DIR, save_name)
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=False, path=save_path
    )

    print(f"\n[Fold {fold}] Starting training for {model_type.upper()}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model for verification or return
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_all_models():
    """
    Orchestrates the training of all models across all folds.
    """
    model_types = ["effnet", "swin"]

    for model_type in model_types:
        for fold in range(Config.N_FOLDS):
            run_fold(fold, model_type)

            # Clear memory
            torch.cuda.empty_cache()
