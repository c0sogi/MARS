import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import pandas as pd
import numpy as np
import os

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    get_class_weights,
    check_initial_loss,
)
from library.data import get_loaders, get_test_loader


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Initializes with ImageNet weights and replaces the head for 4-class classification.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(AppleResNet34, self).__init__()

        # Load pretrained weights if requested
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet34(weights=weights)

        # Replace the fully connected layer
        # The default ResNet34 fc layer has 512 input features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # targets are one-hot encoded or probabilities.
        # CrossEntropyLoss expects class indices for hard classification.
        loss = criterion(outputs, torch.argmax(targets, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Validates the model on the OOB set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, torch.argmax(targets, dim=1))

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    val_auc = calculate_roc_auc(all_targets, all_preds)

    return val_loss, val_auc


def run_training_pipeline():
    """
    Orchestrates the Stratified Bagging training process.
    Trains NUM_BAGS models using bootstrapped datasets.
    """
    seed_everything(Config.SEED)
    Config.setup()

    device = Config.DEVICE
    print(f"Starting training pipeline on device: {device}")

    # Load metadata to calculate class weights
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    class_weights = get_class_weights(full_df, target_cols).to(device)
    print(f"Computed Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Iterate through each seed
    for seed_idx in range(Config.NUM_SEEDS):
        print(f"\n{'='*20} Processing Seed {seed_idx+1}/{Config.NUM_SEEDS} {'='*20}")

        # Seed everything for this run
        seed_everything(Config.SEED + seed_idx)

        train_loader, val_loader = get_loaders()

        model = AppleResNet34(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        model.to(device)

        # Perform initial loss check on the first seed to verify setup
        if seed_idx == 0:
            check_initial_loss(model, train_loader, criterion, device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(Config.MODELS_DIR, f"seed_{seed_idx}_best.pth")

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            # Save best model based on Validation AUC
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New Best Model Saved! AUC: {best_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

    print("\nAll seeds processed.")


def generate_submission_file():
    """
    Generates the submission file by ensembling predictions from all seed models.
    """
    print("\nGenerating submission...")
    device = Config.DEVICE
    test_loader = get_test_loader()

    # Load all trained models
    models_list = []
    for seed_idx in range(Config.NUM_SEEDS):
        model_path = os.path.join(Config.MODELS_DIR, f"seed_{seed_idx}_best.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed_idx} not found at {model_path}. Skipping."
            )
            continue

        # We don't need pretrained weights for inference loading
        model = AppleResNet34(num_classes=Config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models_list.append(model)

    if not models_list:
        print("Error: No models loaded. Cannot generate submission.")
        return

    all_preds = []

    # Get image IDs from test metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    image_ids = test_df["image_id"].values

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Ensemble prediction: Average probabilities across all models
            batch_preds = []
            for model in models_list:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                batch_preds.append(probs.cpu().numpy())

            # Mean across the ensemble dimension (axis 0)
            avg_preds = np.mean(batch_preds, axis=0)
            all_preds.append(avg_preds)

    all_preds = np.concatenate(all_preds)

    # Create submission DataFrame
    # Columns must match the sample submission format
    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    submission_df = pd.DataFrame(all_preds, columns=cols)
    submission_df.insert(0, "image_id", image_ids)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
