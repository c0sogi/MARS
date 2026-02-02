import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data import get_dataloaders


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with an Asymmetric Grouped Convolutional Stem.

    This architecture modifies the first layer to accept 12 channels (4 modalities * 3 slabs)
    using grouped convolutions. The pre-trained ImageNet weights are distributed
    across the groups to ensure each modality is processed by distinct, high-quality filters.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        in_channels=Config.IN_CHANNELS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super().__init__()

        # Load backbone with specified dropout for the head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

        # Modify Stem for Asymmetric Grouped Convolution
        # Original Stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_stem = self.backbone.conv_stem
        out_channels = old_stem.out_channels
        kernel_size = old_stem.kernel_size
        stride = old_stem.stride
        padding = old_stem.padding
        bias = old_stem.bias is not None

        # We use groups=4 to separate the 4 modalities (FLAIR, T1w, T1wCE, T2w)
        # Input channels (12) / Groups (4) = 3 channels per group.
        # This matches the original RGB kernel depth.
        groups = 4

        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        # Asymmetric Initialization
        # The weight shape for both the original and new grouped conv is (32, 3, 3, 3).
        # By copying the weights, we assign the first 8 filters (originally RGB) to Group 1 (FLAIR),
        # the next 8 to Group 2 (T1w), etc. This ensures diversity without symmetry.
        with torch.no_grad():
            new_stem.weight.data = old_stem.weight.data.clone()
            if bias:
                new_stem.bias.data = old_stem.bias.data.clone()

        self.backbone.conv_stem = new_stem

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect predictions for AUC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # robust AUC calculation
    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    """Runs validation loop."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return epoch_loss, auc


def generate_submission(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Saves the result to submission.csv.
    """
    model.eval()
    all_probs = []

    print("Generating predictions with TTA (Original + HFlip + VFlip)...")

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)  # (B, 12, H, W)

            # 1. Original
            out = model(images)
            probs = torch.sigmoid(out)

            # 2. Horizontal Flip (dim 3: W)
            images_h = torch.flip(images, [3])
            out_h = model(images_h)
            probs_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2: H)
            images_v = torch.flip(images, [2])
            out_v = model(images_v)
            probs_v = torch.sigmoid(out_v)

            # Average predictions
            avg_probs = (probs + probs_h + probs_v) / 3.0
            all_probs.extend(avg_probs.cpu().numpy().flatten())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main entry point for the pipeline.
    1. Loads data (with caching).
    2. Initializes model, optimizer, and scheduler.
    3. Runs training loop with Early Stopping.
    4. Loads best model and generates submission.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 3. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.10f}, Train AUC: {train_auc:.10f}, "
            f"Val Loss: {val_loss:.10f}, Val AUC: {val_auc:.10f}"
        )

        scheduler.step(val_auc)

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 5. Inference
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH} for inference...")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    generate_submission(model, test_loader, test_ids, device)
