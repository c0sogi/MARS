import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_logger

logger = get_logger()


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Used here to learn the importance of different slices in the interleaved volume.
    """

    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class InterleavedGroupedStem(nn.Module):
    """
    Specialized input stem for Interleaved Slice-Grouped 2.5D Network.
    Processes 128 channels (32 slices * 4 modalities) using grouped convolutions
    to maintain slice-specific modality fusion before depth aggregation.
    """

    def __init__(self):
        super(InterleavedGroupedStem, self).__init__()

        # Layer 1: Intra-Slice Multi-Modal Fusion
        # Input: 128 channels. Groups=32 means 32 groups of 4 channels.
        # Each group corresponds to 1 slice (FLAIR, T1w, T1wCE, T2w).
        self.conv1 = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.STEM_OUT_CHANNELS,
            kernel_size=3,
            padding=1,
            groups=Config.STEM_GROUPS,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(Config.STEM_OUT_CHANNELS)
        self.act1 = nn.ReLU(inplace=True)

        # Layer 2: Slice Attention (Squeeze-and-Excitation)
        self.se = SqueezeExcite(Config.STEM_OUT_CHANNELS, reduction=16)

        # Layer 3: Depth Aggregation
        # Compresses 128 slice-features down to 64 for the backbone
        self.conv2 = nn.Conv2d(
            in_channels=Config.STEM_OUT_CHANNELS,
            out_channels=Config.BACKBONE_IN_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(Config.BACKBONE_IN_CHANNELS)
        self.act2 = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        # Kaiming/He Normal initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class MGMT25DModel(nn.Module):
    """
    2.5D Convolutional Neural Network for MGMT promoter methylation prediction.
    Combines InterleavedGroupedStem with an EfficientNet-B0 backbone.
    """

    def __init__(self):
        super(MGMT25DModel, self).__init__()
        self.stem = InterleavedGroupedStem()

        # Backbone: EfficientNet-B0
        # in_chans=64 matches the output of the stem
        # num_classes=1 creates the global pooling and final FC layer for binary classification
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=Config.BACKBONE_IN_CHANNELS,
            num_classes=1,
        )

    def forward(self, x):
        # x shape: (B, 128, 256, 256)
        x = self.stem(x)
        # x shape: (B, 64, 256, 256)
        logits = self.backbone(x)
        # logits shape: (B, 1)
        return logits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(train_loader, val_loader):
    """
    Executes the training pipeline with Early Stopping.
    """
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    model = MGMT25DModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_model_path


def predict_and_submit(test_loader, model_path):
    """
    Loads the best model, runs inference on test set, and generates submission.csv.
    """
    device = Config.DEVICE
    model = MGMT25DModel().to(device)

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    ids = []

    logger.info("Starting inference...")
    with torch.no_grad():
        for inputs, pids in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # pids is a tuple/tensor of IDs
            if isinstance(pids, torch.Tensor):
                pids = pids.numpy()

            predictions.extend(probs)
            ids.extend(pids)

    # Create submission DataFrame
    # BraTS21ID should be formatted as string or int depending on sample_submission
    # Sample submission has IDs like 356, 140 (int)
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure BraTS21ID is integer (removing leading zeros if they exist in strings)
    submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
