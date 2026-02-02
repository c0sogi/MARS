import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_roc_auc, get_class_weights


# ====================================================
# Pooling Layer
# ====================================================
class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the p-th power mean of the input feature map.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN in power
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


# ====================================================
# Model Architecture
# ====================================================
class AppleNet(nn.Module):
    """
    Dual-Backbone Heterogeneous Ensemble component.
    Uses a timm backbone with Multi-Level GeM Pooling.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        super(AppleNet, self).__init__()

        # Load backbone with features_only=True to access intermediate layers
        # We extract features from the last 3 reduction stages (indices 2, 3, 4)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4)
        )

        # Get the channel dimensions for the extracted features
        feature_channels = self.backbone.feature_info.channels()

        # Create a GeM pooling layer for each feature level
        self.gem_layers = nn.ModuleList([GeM(p=Config.GEM_P) for _ in feature_channels])

        # Calculate total concatenated feature dimension
        total_features = sum(feature_channels)

        # Linear classifier (Backbone -> Pooling -> Linear)
        self.fc = nn.Linear(total_features, num_classes)

    def forward(self, x):
        # Extract features (returns a list of tensors)
        features = self.backbone(x)

        pooled_features = []
        for i, feat in enumerate(features):
            # Apply GeM pooling: (B, C, H, W) -> (B, C, 1, 1)
            pooled = self.gem_layers[i](feat)
            # Flatten: (B, C, 1, 1) -> (B, C)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate features from all levels
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        logits = self.fc(concat_features)
        return logits


# ====================================================
# Model EMA
# ====================================================
class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a shadow copy of the model that is updated using EMA.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        if device:
            self.ema.to(device)

        # Disable gradients for EMA model
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            # Update parameters
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    esd[k].copy_(esd[k] * self.decay + msd[k] * (1.0 - self.decay))
                else:
                    # Copy non-floating point parameters (e.g. num_batches_tracked)
                    esd[k].copy_(msd[k])


# ====================================================
# Training & Evaluation Helper Functions
# ====================================================
def train_one_epoch(model, loader, optimizer, criterion, device, ema=None):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema:
            ema.update(model)

        running_loss += loss.item() * images.size(0)

        # Store predictions for AUC calculation
        probs = torch.softmax(logits, dim=1)
        all_targets.append(targets.detach().cpu())
        all_preds.append(probs.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Convert one-hot targets to class indices for ROC AUC if needed,
    # but calculate_roc_auc handles (N, C) inputs.
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)
            all_targets.append(targets.detach().cpu())
            all_preds.append(probs.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def train_fold(model_name, train_loader, val_loader, train_df, fold_idx=0):
    """
    Trains a single model (backbone) for a specific fold.
    Returns the path to the best saved model.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = AppleNet(
        model_name=model_name, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    # Initialize EMA
    ema = (
        ModelEMA(model, decay=Config.EMA_DECAY, device=device)
        if Config.USE_EMA
        else None
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Loss Function with Class Weights
    class_weights = get_class_weights(train_df, Config.CLASS_LABELS)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"{model_name}_fold_{fold_idx}.pth"
    )

    print(f"\nStarting training for {model_name} - Fold {fold_idx}")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema
        )

        # Validate using EMA model if available, else standard model
        eval_model = ema.ema if ema else model
        val_loss, val_auc = validate(eval_model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(eval_model.state_dict(), best_model_path)
            patience_counter = 0
            # print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_model_path, best_auc


# ====================================================
# Inference & Submission
# ====================================================
def predict_and_submit(models_paths, test_loader, output_path):
    """
    Generates predictions using an ensemble of models and saves the submission file.
    Implements TTA (Horizontal Flip).
    """
    device = torch.device(Config.DEVICE)
    print(f"\nGenerating predictions with {len(models_paths)} models...")

    # Placeholder for ensemble predictions
    # Shape: (Num_Test_Images, Num_Classes)
    ensemble_preds = None
    image_ids = test_loader.dataset.df["image_id"].values

    for model_path in models_paths:
        # Determine model name from path (heuristic based on naming convention)
        # We need to instantiate the correct architecture.
        # Assuming path format: ".../model_name_fold_X.pth"
        filename = os.path.basename(model_path)
        # Extract model name by removing _fold_X.pth
        # This is a bit fragile, so we iterate Config.BACKBONES to find the match
        arch_name = None
        for b in Config.BACKBONES:
            if b in filename:
                arch_name = b
                break

        if arch_name is None:
            print(
                f"Warning: Could not determine architecture for {filename}. Skipping."
            )
            continue

        # Load Model
        model = AppleNet(
            model_name=arch_name, num_classes=Config.NUM_CLASSES, pretrained=False
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # 1. Original Prediction
                with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)

                # 2. TTA: Horizontal Flip
                if Config.TTA_FLIP_HORIZONTAL:
                    images_flipped = torch.flip(
                        images, dims=[3]
                    )  # [B, C, H, W], flip W
                    with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                        logits_flipped = model(images_flipped)
                        probs_flipped = torch.softmax(logits_flipped, dim=1)

                    # Average Original and TTA
                    probs = (probs + probs_flipped) / 2.0

                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)

        if ensemble_preds is None:
            ensemble_preds = fold_preds
        else:
            ensemble_preds += fold_preds

    # Average over all models
    ensemble_preds /= len(models_paths)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(ensemble_preds, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", image_ids)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
