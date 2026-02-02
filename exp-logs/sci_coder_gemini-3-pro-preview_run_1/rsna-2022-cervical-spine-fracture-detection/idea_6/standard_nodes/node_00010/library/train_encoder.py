import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from library import config, models, datasets, utils


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Stage2Wrapper(nn.Module):
    """
    Wraps the MaskedCNNEncoder with a classification head for training.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = models.MaskedCNNEncoder(pretrained=pretrained)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(config.STAGE2_CONFIG["feature_dim"], 1)

    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate simple train metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        train_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        train_auc = 0.5

    return epoch_loss, train_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Binary predictions for accuracy (threshold 0.5)
    binary_preds = (all_preds > 0.5).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, acc, auc


def run_stage2_training(
    epochs=config.STAGE2_CONFIG["epochs"],
    batch_size=config.STAGE2_CONFIG["batch_size"],
    lr=config.STAGE2_CONFIG["lr"],
    patience=3,
):
    set_seed()

    print(f"Starting Stage 2 Training: Mask-Conditioned Feature Encoder")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # 1. Datasets
    train_ds, val_ds = datasets.get_datasets(stage="stage2")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model
    # We wrap the encoder to add a classification head
    model = Stage2Wrapper(pretrained=True)
    model = model.to(config.DEVICE)

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Unweighted BCE as per instructions
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "stage2_encoder.pth")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )

        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, config.DEVICE
        )

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {val_acc}")
        print(f"Val AUC: {val_auc}")

        # Scheduler step based on Val Loss
        scheduler.step(val_loss)

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation Loss improved from {best_val_loss} to {val_loss}. Saving encoder..."
            )
            best_val_loss = val_loss
            # Save only the encoder part, as Stage 3 only needs the feature extractor
            torch.save(model.encoder.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Stage 2 Training Completed. Best Val Loss: {best_val_loss}")
    print(f"Encoder model saved to: {checkpoint_path}")
