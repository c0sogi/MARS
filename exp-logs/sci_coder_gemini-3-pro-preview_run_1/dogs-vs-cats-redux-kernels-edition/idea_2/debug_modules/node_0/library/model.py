import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    save_submission,
)
from library.dataset import DogCatDataset, get_transforms


class DogCatClassifier(nn.Module):
    """
    Dog vs Cat Classifier using EfficientNetV2-M backbone.
    Replaces the default head with GAP + Linear layer.
    """

    def __init__(
        self,
        model_name=Config.backbone,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
    ):
        super(DogCatClassifier, self).__init__()
        # Load backbone with global average pooling enabled (num_classes=0 removes default head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            global_pool="avg",
        )

        # Get the feature dimension of the backbone
        in_features = self.backbone.num_features

        # Define the simple linear classification head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Extract features: (Batch_Size, Num_Features)
        features = self.backbone(x)

        # Classification logits: (Batch_Size, Num_Classes)
        logits = self.fc(features)

        # Squeeze to (Batch_Size,) for binary classification compatibility
        return logits.squeeze(1)


def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, epoch):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def predict_test(test_loader, model, device, use_tta=Config.use_tta):
    """
    Generates predictions for the test set.
    Implements Test Time Augmentation (TTA) if enabled.
    """
    model.eval()
    preds = []
    ids_list = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            ids_list.extend(ids.numpy())

            with autocast():
                # Forward pass 1: Original Image
                out1 = torch.sigmoid(model(images))

                if use_tta:
                    # Forward pass 2: Horizontally Flipped Image
                    # dim=3 is width in (N, C, H, W)
                    images_flipped = torch.flip(images, dims=[3])
                    out2 = torch.sigmoid(model(images_flipped))

                    # Average probabilities
                    out = (out1 + out2) / 2.0
                else:
                    out = out1

            preds.extend(out.cpu().numpy())

    return np.array(ids_list), np.array(preds)


def run_training(debug=Config.debug):
    """
    Main execution function.
    Runs 5-Fold Cross-Validation training and generates the final submission.
    """
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # Load and merge train and validation metadata to use 100% of labeled data for CV
    df_train_part = pd.read_csv(Config.train_metadata_path)
    df_val_part = pd.read_csv(Config.val_metadata_path)
    df_train_full = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    df_test = pd.read_csv(Config.test_metadata_path)

    if debug:
        print("DEBUG MODE: Using subset of data.")
        df_train_full = df_train_full.sample(
            n=200, random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(n=50, random_state=Config.seed).reset_index(drop=True)

    # Prepare Test Loader (Fixed for all folds)
    test_dataset = DogCatDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # --- 5-Fold Stratified Cross-Validation ---
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Array to accumulate test predictions from each fold
    # Initialize with zeros
    fold_test_preds = np.zeros(len(df_test))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        print(f"\n{'='*20} Fold {fold + 1}/{Config.n_folds} {'='*20}")

        # Split Data for this fold
        df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = DogCatDataset(
            df_train, transforms=get_transforms("train"), mode="train"
        )
        val_dataset = DogCatDataset(
            df_val, transforms=get_transforms("valid"), mode="val"
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model, Criterion, Optimizer, Scheduler
        model = DogCatClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.epochs, eta_min=Config.min_lr
        )
        scaler = GradScaler()

        best_val_loss = float("inf")
        best_model_path = os.path.join(Config.model_dir, f"model_fold_{fold+1}.pth")

        # Training Loop
        for epoch in range(Config.epochs):
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, device, scaler, epoch
            )
            val_loss = validate(val_loader, model, criterion, device)
            scheduler.step()

            print(
                f"Fold {fold+1} | Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {"state_dict": model.state_dict(), "val_loss": val_loss},
                    is_best=True,
                    filename=f"model_fold_{fold+1}.pth",
                    folder=Config.model_dir,
                )

        print(f"Fold {fold+1} Finished. Best Val Loss: {best_val_loss:.6f}")

        # --- Inference for this Fold ---
        # Load best weights
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict
        ids, preds = predict_test(test_loader, model, device)
        fold_test_preds += preds

        # Cleanup to free memory
        del model, optimizer, scaler, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # --- Ensemble Aggregation ---
    # Average predictions across all folds
    avg_preds = fold_test_preds / Config.n_folds

    # --- Submission ---
    output_path = os.path.join(Config.submission_dir, "submission.csv")
    save_submission(ids, avg_preds, output_path=output_path)
    print(f"\nEnsemble prediction complete. Submission saved to {output_path}")
