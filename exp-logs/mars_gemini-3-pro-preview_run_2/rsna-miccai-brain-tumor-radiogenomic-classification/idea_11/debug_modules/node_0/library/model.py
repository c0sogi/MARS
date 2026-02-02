import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import timm

from library.config import Config
from library.data import get_dataloader


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Asymmetric Grouped Convolutional Stem.

    - Input: 12 channels (4 modalities * 3 slices)
    - Stem: Grouped Convolution (groups=4) to isolate modalities.
    - Initialization: Asymmetric Filter Distribution (distributing ImageNet weights across groups).
    - Head: Regularized with Dropout.
    """

    def __init__(self):
        super().__init__()

        # Load backbone
        # tf_efficientnet_b0_ns: EfficientNet-B0 No-Noisy-Student (ImageNet pre-trained)
        # num_classes=0 removes the classifier
        # global_pool="" gives us the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, global_pool=""
        )

        # ----------------------------------------------------------------------
        # 1. Modify Stem for 12-Channel Input & Grouped Convolution
        # ----------------------------------------------------------------------
        original_stem = self.backbone.conv_stem

        # New Stem Configuration
        # in_channels=12 (4 modalities * 3 slices)
        # out_channels=32 (Standard B0 stem output)
        # groups=4 (One group per modality)
        self.new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
            groups=Config.GROUPS,
        )

        # ----------------------------------------------------------------------
        # 2. Asymmetric Filter Initialization
        # ----------------------------------------------------------------------
        self._init_asymmetric_weights(original_stem, self.new_stem)

        # Replace the stem in the backbone
        self.backbone.conv_stem = self.new_stem

        # ----------------------------------------------------------------------
        # 3. Regularized Classification Head
        # ----------------------------------------------------------------------
        self.num_features = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.num_features, Config.NUM_CLASSES),
        )

    def _init_asymmetric_weights(self, original_layer, new_layer):
        """
        Distributes the full bank of 32 pre-trained ImageNet filters across the
        4 modality groups.

        Original Weights: (32, 3, 3, 3)
        New Weights (groups=4): (32, 3, 3, 3)  [Since 12 input / 4 groups = 3]

        By directly copying the weights, we achieve the asymmetric distribution:
        - Filters 0-7 (originally RGB) -> Group 0 (FLAIR)
        - Filters 8-15 (originally RGB) -> Group 1 (T1w)
        - etc.
        """
        with torch.no_grad():
            new_layer.weight.data.copy_(original_layer.weight.data)
            if original_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.data.copy_(original_layer.bias.data)

    def forward(self, x):
        # x: (B, 12, H, W)

        # Backbone extraction
        x = self.backbone(x)  # (B, 1280, H/32, W/32)

        # Pooling
        x = self.global_pool(x)  # (B, 1280, 1, 1)
        x = x.flatten(1)  # (B, 1280)

        # Classification Head
        logits = self.head(x)  # (B, 1)
        return logits


# ------------------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

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
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle case where batch might contain only one class causing ROC error
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    print(f"Initializing AsymmetricEfficientNet on {Config.DEVICE}...")
    model = AsymmetricEfficientNet().to(Config.DEVICE)

    # Signal-Preserving Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = get_dataloader("train", debug=Config.DEBUG)
    val_loader = get_dataloader("val", debug=Config.DEBUG)

    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f} - "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"  [New Best Model] Saved to {Config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc:.6f}")
    return model


# ------------------------------------------------------------------------------
# Inference Logic
# ------------------------------------------------------------------------------
def predict_and_submit():
    print("Starting Inference with TTA...")

    # Load Best Model
    model = AsymmetricEfficientNet().to(Config.DEVICE)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded weights from {Config.CHECKPOINT_PATH}")
    else:
        print("Warning: Checkpoint not found. Using random weights (for debugging).")

    model.eval()

    test_loader = get_dataloader(
        "test", debug=Config.DEBUG, batch_size=Config.BATCH_SIZE
    )

    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            # images shape: (B, 12, H, W)
            images = images.to(Config.DEVICE)

            # TTA: Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA: Horizontal Flip
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # TTA: Vertical Flip
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average Predictions
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            predictions.extend(avg_probs.cpu().numpy().flatten())

    # Load test metadata to get IDs
    df_test = pd.read_csv(Config.TEST_METADATA)
    if Config.DEBUG:
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Ensure lengths match
    if len(predictions) != len(df_test):
        print(
            f"Warning: Prediction count {len(predictions)} != Test ID count {len(df_test)}"
        )
        min_len = min(len(predictions), len(df_test))
        predictions = predictions[:min_len]
        df_test = df_test.iloc[:min_len]

    submission = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    run_training()
    predict_and_submit()
