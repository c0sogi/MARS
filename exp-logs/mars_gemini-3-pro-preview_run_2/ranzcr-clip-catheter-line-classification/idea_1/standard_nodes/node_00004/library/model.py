import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.dataset import get_dataloaders


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CatheterModel(nn.Module):
    """
    ResNet-34 based model for Catheter and Line Detection.
    Replaces the final fully connected layer to output logits for 11 classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(CatheterModel, self).__init__()
        # Load pre-trained ResNet-34
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Replace the final fully connected layer
        # ResNet34 structure: ... -> avgpool -> fc
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    save_path=None,
):
    """
    Trains the CatheterModel.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        debug (bool): Whether to run in debug mode (subset of data).
        save_path (str): Path to save the best model checkpoint.
    """
    set_seed()

    device = Config.DEVICE
    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # DataLoaders
    loaders = get_dataloaders(batch_size=batch_size, debug=debug)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model
    model = CatheterModel(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    # Mixed Precision
    scaler = GradScaler()

    best_auc = 0.0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                # Apply sigmoid for metric calculation
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate AUC per column
        aucs = []
        for i in range(Config.NUM_CLASSES):
            # Check if class exists in validation set to avoid error
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            else:
                # If only one class is present, AUC is undefined.
                # We can skip or assume 0.5. Skipping is safer for average.
                pass

        avg_auc = np.mean(aucs) if aucs else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val AUC: {avg_auc:.10f}"
        )

        # Save best model
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {avg_auc:.10f}")

    return best_auc


def predict_and_submit(
    model_path=None, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Runs inference on the test set and creates the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        batch_size (int): Batch size.
        debug (bool): Debug mode.
    """
    device = Config.DEVICE
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Loaders
    loaders = get_dataloaders(batch_size=batch_size, debug=debug)
    test_loader = loaders["test"]

    # Model
    model = CatheterModel(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_uids = []

    print("Starting inference...")

    with torch.no_grad():
        for images, uids in test_loader:
            images = images.to(device)

            # Forward pass
            # No autocast needed strictly for inference but can speed up
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_uids.extend(uids)

    all_probs = np.concatenate(all_probs)

    # Create Submission DataFrame
    # Columns must be StudyInstanceUID followed by targets
    submission_data = {"StudyInstanceUID": all_uids}

    for i, col_name in enumerate(Config.TARGET_COLS):
        submission_data[col_name] = all_probs[:, i]

    df_sub = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())
