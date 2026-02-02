import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import os
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm

# Import provided library modules
from library import config
from library import dataset


class EfficientNetB4Native(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(EfficientNetB4Native, self).__init__()

        # Load Pretrained Backbone
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Feature dimension for EfficientNet-B4 is 1792
        # We access the classifier to get the input features size, then discard it
        self.feature_dim = self.backbone.classifier[1].in_features

        # We do not use the original classifier or avgpool
        # We will use the 'features' part of the backbone

        # Custom Head: Concatenated Global Pooling (GAP + GMP) -> Linear
        # Input dim is feature_dim * 2 because of concatenation
        self.head = nn.Linear(self.feature_dim * 2, num_classes)

    def forward(self, x):
        # Extract features: (B, 1792, H, W)
        x = self.backbone.features(x)

        # Global Average Pooling: (B, 1792)
        x_avg = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Global Max Pooling: (B, 1792)
        x_max = F.adaptive_max_pool2d(x, 1).flatten(1)

        # Concatenate: (B, 3584)
        x_cat = torch.cat([x_avg, x_max], dim=1)

        # Classification
        logits = self.head(x_cat)
        return logits

    def freeze_backbone(self):
        """Freezes all parameters in the backbone, keeps head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Ensure head is trainable
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_blocks(self, n_blocks):
        """
        Unfreezes the top n_blocks of the backbone features.
        The 'features' attribute is a Sequential container.
        """
        # First, ensure head is trainable
        for param in self.head.parameters():
            param.requires_grad = True

        # Get list of feature blocks
        # efficientnet_b4.features contains 9 main modules (0-8)
        feature_layers = list(self.backbone.features.children())
        total_layers = len(feature_layers)

        # Determine split point
        start_idx = max(0, total_layers - n_blocks)

        print(
            f"Unfreezing backbone blocks from index {start_idx} to {total_layers - 1}..."
        )

        # Iterate and unfreeze
        for i, layer in enumerate(feature_layers):
            if i >= start_idx:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                # Ensure lower layers remain frozen
                for param in layer.parameters():
                    param.requires_grad = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect for metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_f1


def predict(model, loader, device):
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            ids.extend(batch_ids)

    return ids, predictions


def run_training():
    print(f"Initializing training for {config.PROJECT_NAME}...")
    config.seed_everything(config.SEED)

    # 1. Data Setup
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # 2. Model Setup
    model = EfficientNetB4Native(
        num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED
    )
    model = model.to(config.DEVICE)

    # 3. Loss Setup (Weighted)
    if config.USE_CLASS_WEIGHTS:
        class_weights = dataset.calculate_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # ====================================================
    # Stage 1: Train Head Only (Frozen Backbone)
    # ====================================================
    print("\n=== Stage 1: Training Head (Backbone Frozen) ===")
    model.freeze_backbone()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_STAGE1,
    )

    for epoch in range(config.NUM_EPOCHS_STAGE1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE1} | "
            f"Train Loss: {train_loss:.6f} F1: {train_f1:.6f} | "
            f"Val Loss: {val_loss:.6f} F1: {val_f1:.10f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best weights from Stage 1 before starting Stage 2
    model.load_state_dict(best_model_wts)

    # ====================================================
    # Stage 2: Fine-Tuning (Unfreeze Top Blocks)
    # ====================================================
    print("\n=== Stage 2: Fine-Tuning (Unfreezing Top Blocks) ===")
    # Unfreeze top 3 blocks (EfficientNet B4 has ~9 main blocks)
    model.unfreeze_blocks(n_blocks=3)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_STAGE2,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS_STAGE2
    )

    for epoch in range(config.NUM_EPOCHS_STAGE2):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS_STAGE2} | "
            f"Train Loss: {train_loss:.6f} F1: {train_f1:.6f} | "
            f"Val Loss: {val_loss:.6f} F1: {val_f1:.10f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save checkpoint immediately
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print(f"\nTraining Complete. Best Validation F1: {best_f1:.10f}")

    # ====================================================
    # Inference and Submission
    # ====================================================
    print("Generating submission...")

    # Load best model
    model.load_state_dict(best_model_wts)

    # Predict
    test_ids, test_preds = predict(model, test_loader, config.DEVICE)

    # Save
    submission_df = pd.DataFrame({"Id": test_ids, "Predicted": test_preds})

    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
