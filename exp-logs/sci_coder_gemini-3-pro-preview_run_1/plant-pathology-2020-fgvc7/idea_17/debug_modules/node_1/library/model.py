import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    seed_everything,
    get_class_weights,
    calculate_roc_auc,
    save_checkpoint,
)
from library.dataset import AppleDataset, get_transforms


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Initializes with ImageNet weights and uses a simple Global Average Pooling + Linear head.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(AppleResNet34, self).__init__()
        # Initialize ResNet34 backbone
        # Using 'weights' for modern torchvision versions, falling back to 'pretrained' if needed
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        except AttributeError:
            self.backbone = models.resnet34(pretrained=pretrained)

        # Replace the default fully connected layer
        # The default ResNet34 structure ends with: AvgPool -> Flatten -> FC
        # We replace FC with our own Linear layer.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Simple classification head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Backbone forward pass (includes GAP and Flatten due to implementation of resnet34)
        x = self.backbone(x)
        # Classification head
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        all_targets.append(targets.cpu().numpy())
        all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except Exception:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device, use_tta=False):
    """
    Validates the model. Supports Test Time Augmentation (TTA).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Standard forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # Horizontal Flip
                inputs_h = torch.flip(inputs, dims=[3])
                outputs_h = model(inputs_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # Vertical Flip
                inputs_v = torch.flip(inputs, dims=[2])
                outputs_v = model(inputs_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average predictions
                probs = (probs + probs_h + probs_v) / 3.0

            # Calculate loss (using single view for consistency)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = calculate_roc_auc(all_targets, all_preds)
    except Exception:
        auc = 0.5

    return epoch_loss, auc, all_preds, all_targets


def run_training():
    """
    Executes the High-Density Stratified K-Fold (K=10) training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Metadata
    train_df_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df_full = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine for K-Fold splitting
    full_df = pd.concat([train_df_full, val_df_full]).reset_index(drop=True)

    # Ensure stratify label exists
    if "stratify_label" not in full_df.columns:
        full_df["stratify_label"] = full_df[Config.CLASSES].idxmax(axis=1)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Calculate Class Weights
    class_weights = get_class_weights(Config.TRAIN_METADATA_PATH)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Storage for OOF predictions
    oof_preds_single = np.zeros((len(full_df), Config.NUM_CLASSES))
    oof_preds_tta = np.zeros((len(full_df), Config.NUM_CLASSES))
    oof_targets = np.zeros((len(full_df), Config.NUM_CLASSES))

    print(f"Starting {Config.N_FOLDS}-Fold Training on {len(full_df)} images...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["stratify_label"])
    ):
        print(f"\n=== Fold {fold} ===")

        # Split Data
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets and Loaders
        train_ds = AppleDataset(train_df, transforms=get_transforms("train"))
        val_ds = AppleDataset(val_df, transforms=get_transforms("valid"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = AppleResNet34(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        model.to(device)

        # Optimizer and Scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.EPOCHS, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        # Initial Loss Test (Sanity Check)
        model.eval()
        with torch.no_grad():
            init_inputs, init_targets = next(iter(train_loader))
            init_inputs, init_targets = init_inputs.to(device), init_targets.to(device)
            init_outputs = model(init_inputs)
            init_loss = criterion(init_outputs, init_targets).item()
            # print(f"  Initial Loss: {init_loss:.4f}")

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc, _, _ = validate(
                model, val_loader, criterion, device, use_tta=False
            )

            scheduler.step()

            print(
                f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, best_model_path, best_auc)

        # Load Best Model for OOF Generation
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate OOF Predictions (Single View)
        _, _, preds_single, targets = validate(
            model, val_loader, criterion, device, use_tta=False
        )
        oof_preds_single[val_idx] = preds_single
        oof_targets[val_idx] = targets

        # Generate OOF Predictions (TTA View)
        _, _, preds_tta, _ = validate(
            model, val_loader, criterion, device, use_tta=True
        )
        oof_preds_tta[val_idx] = preds_tta

    # Calculate Overall OOF Metrics
    auc_single = calculate_roc_auc(oof_targets, oof_preds_single)
    auc_tta = calculate_roc_auc(oof_targets, oof_preds_tta)

    print(f"\nOverall OOF AUC (Single View): {auc_single:.6f}")
    print(f"Overall OOF AUC (TTA View):    {auc_tta:.6f}")

    # Decision Rule for TTA
    use_tta = auc_tta > auc_single
    print(f"Decision: Use TTA for Inference? {use_tta}")

    return use_tta


def predict_test(use_tta: bool):
    """
    Generates predictions for the test set using the ensemble of trained models.
    """
    device = torch.device(Config.DEVICE)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Test Dataset
    test_ds = AppleDataset(
        test_df, transforms=get_transforms("test"), output_label=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    final_preds = np.zeros((len(test_df), Config.NUM_CLASSES))
    models_used = 0

    print("\nGenerating Test Predictions...")

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found at {model_path}")
            continue

        # Load Model
        model = AppleResNet34(num_classes=Config.NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                if use_tta:
                    # Horizontal Flip
                    inputs_h = torch.flip(inputs, dims=[3])
                    outputs_h = model(inputs_h)
                    probs_h = torch.softmax(outputs_h, dim=1)

                    # Vertical Flip
                    inputs_v = torch.flip(inputs, dims=[2])
                    outputs_v = model(inputs_v)
                    probs_v = torch.softmax(outputs_v, dim=1)

                    # Average
                    probs = (probs + probs_h + probs_v) / 3.0

                fold_preds.append(probs.cpu().numpy())

        final_preds += np.concatenate(fold_preds)
        models_used += 1

    if models_used > 0:
        final_preds /= models_used

    # Save Submission
    submission_df = pd.DataFrame(final_preds, columns=Config.CLASSES)
    submission_df.insert(0, "image_id", test_df["image_id"])
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    """
    Main execution pipeline.
    """
    # 1. Run Training (K-Fold CV)
    use_tta = run_training()

    # 2. Generate Submission
    predict_test(use_tta)
