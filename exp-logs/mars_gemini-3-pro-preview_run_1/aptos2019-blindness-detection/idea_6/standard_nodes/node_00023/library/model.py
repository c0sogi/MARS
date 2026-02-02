import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa


class EfficientNetV2Ordinal(nn.Module):
    """
    EfficientNetV2-Small with a Rank-Consistent Ordinal Regression Head.
    """

    def __init__(
        self, model_name=Config.MODEL_NAME, pretrained=True, drop_rate=Config.DROP_RATE
    ):
        super(EfficientNetV2Ordinal, self).__init__()

        # Load backbone with num_classes=0 to get pooled features (Global Average Pooling)
        # This removes the original classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            in_features = self.backbone.get_classifier().in_features

        # Ordinal Regression Head
        # We use 4 binary output units to represent 5 classes (0-4)
        # 0: [0,0,0,0], 1: [1,0,0,0], 2: [1,1,0,0], 3: [1,1,1,0], 4: [1,1,1,1]
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate), nn.Linear(in_features, Config.NUM_OUTPUTS)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Calculate loss
        # BCEWithLogitsLoss with reduction='none' returns (Batch, 4)
        # We sum over the 4 units (dim=1) to get the total loss per sample
        # Then we average over the batch
        loss_per_sample = criterion(logits, targets).sum(dim=1)
        loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Validates the model and calculates QWK score.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            # Loss calculation
            loss_per_sample = criterion(logits, targets).sum(dim=1)
            loss = loss_per_sample.mean()
            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Predictions for QWK
            # Strategy: Sigmoid -> Sum Probabilities -> Round to nearest int
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            all_preds.append(scores.cpu().numpy())

            # Targets for QWK
            # Reconstruct integer class from ordinal vector by summing
            true_labels = targets.sum(dim=1)
            all_targets.append(true_labels.cpu().numpy())

    val_loss = running_loss / count
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(all_targets, all_preds)

    return val_loss, qwk


def run_training(
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    save_path=Config.BEST_MODEL_PATH,
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = EfficientNetV2Ordinal()
    model = model.to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # Loss Function
    # reduction='none' allows us to sum across the ordinal units manually
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_qwk = -np.inf
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val QWK: {val_qwk}"
        )

        # Save best model based on QWK
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best QWK: {best_qwk}")
    return best_qwk


def generate_submission(
    test_loader, model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = EfficientNetV2Ordinal()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights."
        )

    model = model.to(device)
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for images, id_codes in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)

            # Round to nearest integer for final class prediction
            preds = torch.round(scores).long()

            # Clip to ensure valid range 0-4
            preds = torch.clamp(preds, 0, 4)

            ids.extend(id_codes)
            predictions.extend(preds.cpu().numpy())

    # Create Submission DataFrame
    df = pd.DataFrame({"id_code": ids, "diagnosis": predictions})

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
