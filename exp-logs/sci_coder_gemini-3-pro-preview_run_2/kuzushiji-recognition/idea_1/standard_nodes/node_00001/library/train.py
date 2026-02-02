import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, seed_everything
from library.models import SegmentationUNet, CharacterClassifier
from library.data import get_dataloaders


class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
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
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_segmenter(debug=False, epochs=Config.SEG_EPOCHS):
    """
    Trains the Segmentation Model (Stage 1).
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Segmentation Training on {device}...")

    # 1. Data
    train_loader = get_dataloaders("segmentation", "train", debug=debug)
    val_loader = get_dataloaders("segmentation", "val", debug=debug)

    # 2. Model
    model = SegmentationUNet(n_classes=1).to(device)

    # 3. Optimization
    # Combination of BCE and Dice Loss
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.SEG_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    save_path = os.path.join(Config.CACHE_DIR, "seg_model.pth")
    early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)

            optimizer.zero_grad()
            logits = model(images)

            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).float().unsqueeze(1)

                logits = model(images)

                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice

                val_loss += loss.item() * images.size(0)

                # Calculate Dice Score for metrics (1 - DiceLoss)
                val_dice_score += (1 - loss_dice.item()) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice_score /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice Score: {val_dice_score}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Segmentation training complete.")
    return save_path


def train_classifier(debug=False, epochs=Config.CLS_EPOCHS):
    """
    Trains the Classification Model (Stage 2).
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Classification Training on {device}...")

    # 1. Data
    # Note: ClassificationDataset handles caching internally
    train_loader = get_dataloaders("classification", "train", debug=debug)
    val_loader = get_dataloaders("classification", "val", debug=debug)

    # Determine number of classes from the dataset
    # train_loader.dataset is a ClassificationDataset instance
    num_classes = len(train_loader.dataset.label2id)
    print(f"Number of classes: {num_classes}")

    # 2. Model
    model = CharacterClassifier(num_classes=num_classes).to(device)

    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.CLS_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    save_path = os.path.join(Config.CACHE_DIR, "cls_model.pth")
    early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Classification training complete.")
    return save_path
