import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

from library.config import (
    INPUT_CHANNELS,
    GROUPS,
    DROPOUT_RATE,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    METADATA_TEST,
)
from library.utils import setup_logger, seed_everything, ensure_dir
from library.data_loader import get_dataloaders

logger = setup_logger("model")


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Biologically-Adaptive Input Strategy.

    Modifications:
    1. Input Stem: Accepts 12 channels (4 modalities * 3 slices) via Grouped Convolution (groups=4).
    2. Initialization: Direct Block Copy of ImageNet weights to assign pre-trained filters to modality groups.
    3. Head: Regularized classification head with Dropout (p=0.5).
    """

    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # Load pre-trained backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        self._modify_stem()
        self._modify_head()

    def _modify_stem(self):
        # Access the first Conv2d layer in the features Sequential block
        # efficientnet_b0 structure: features[0] is Conv2dNormActivation, features[0][0] is Conv2d
        old_conv = self.backbone.features[0][0]

        # Create new Conv2d with 12 input channels and groups=4
        # We preserve the original kernel size, stride, and padding (geometry)
        new_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            groups=GROUPS,
            bias=False,  # EfficientNet uses BatchNormalization, so bias is False
        )

        # Direct Block Copy Initialization
        # Original weights shape: (32, 3, 3, 3)
        # New weights shape: (32, 3, 3, 3) [since in_channels / groups = 12 / 4 = 3]
        # We copy the weights directly. This assigns filters 0-7 to Group 1 (FLAIR),
        # filters 8-15 to Group 2 (T2w), etc., without interleaving.
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def _modify_head(self):
        # Replace the classifier block
        # Original is usually Sequential(Dropout, Linear)
        # We enforce our specific Dropout rate and output dimension
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch might contain only one class
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
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    TTA Strategies: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    predictions = []

    # Load test metadata to ensure ID alignment
    # Use the loader's dataset to ensure ID alignment (Cite debug_lesson_7)
    test_ids = loader.dataset.df["BraTS21ID"].tolist()

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip (flip on width dimension, dim=3)
            # Input shape: (B, C, H, W)
            out2 = torch.sigmoid(model(torch.flip(inputs, [3])))

            # 3. Vertical Flip (flip on height dimension, dim=2)
            out3 = torch.sigmoid(model(torch.flip(inputs, [2])))

            # Average predictions
            avg_prob = (out1 + out2 + out3) / 3.0
            predictions.extend(avg_prob.cpu().numpy().flatten())

    return test_ids, predictions


def run(debug=False, max_samples=None):
    """
    Main execution function:
    1. Sets up data and model.
    2. Trains the model with Early Stopping.
    3. Loads the best model.
    4. Generates predictions with TTA.
    5. Saves submission file.
    """
    seed_everything(SEED)
    ensure_dir(os.path.dirname(MODEL_SAVE_PATH))
    ensure_dir(os.path.dirname(SUBMISSION_PATH))

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, max_samples=max_samples
    )

    # 2. Model Initialization
    logger.info("Initializing AsymmetricEfficientNet...")
    model = AsymmetricEfficientNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 3. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Train AUC: {train_auc} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc}"
        )

        # Early Stopping & Model Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"New best model saved with Val AUC: {best_val_auc}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # 4. Inference
    logger.info("Loading best model for inference...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning(
            "No model file found. Using current model state (likely untrained or failed)."
        )

    logger.info("Generating predictions with TTA...")
    test_ids, preds = predict_with_tta(model, test_loader, device)

    # 5. Submission
    logger.info("Saving submission...")
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})

    # Ensure proper formatting (5-digit ID not required for submission CSV based on sample, just int)
    # But sample_submission.csv has BraTS21ID as int.

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")
