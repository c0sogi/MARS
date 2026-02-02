import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score

from library import config, data, utils

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class AsymmetricEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load Pre-trained Backbone
        # We use IMAGENET1K_V1 weights as the foundation
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem for 20-Channel Input with Grouped Convolutions
        # Original Stem: Conv2d(3, 32, kernel=3, stride=2, padding=1, bias=False)
        original_conv = self.backbone.features[0][0]
        original_weights = original_conv.weight.data  # Shape: (32, 3, 3, 3)

        # Configuration for the new stem
        in_channels = config.INPUT_CHANNELS  # 20
        out_channels = original_conv.out_channels  # 32
        groups = config.GROUPS  # 4
        kernel_size = original_conv.kernel_size  # (3, 3)
        stride = original_conv.stride  # (2, 2)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # (1, 1)

        # Create the new surgical layer
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )

        # 3. Asymmetric & Expanded Initialization
        # We need to map 3-channel weights to 5-channel groups.
        # Target shape: (32, 5, 3, 3) because in_channels_per_group = 20/4 = 5.
        # Strategy: Central Replication [W0, W1, W1, W1, W2]

        new_weights = torch.zeros_like(new_conv.weight.data)

        # W0 -> Channel 0 (Edge)
        new_weights[:, 0, :, :] = original_weights[:, 0, :, :]
        # W1 -> Channels 1, 2, 3 (Center/Texture)
        new_weights[:, 1, :, :] = original_weights[:, 1, :, :]
        new_weights[:, 2, :, :] = original_weights[:, 1, :, :]
        new_weights[:, 3, :, :] = original_weights[:, 1, :, :]
        # W2 -> Channel 4 (Edge)
        new_weights[:, 4, :, :] = original_weights[:, 2, :, :]

        # Assign weights
        new_conv.weight.data = new_weights

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 4. Reconstruct Classification Head
        # Original: Dropout -> Linear(1280, 1000)
        # New: Dropout(0.5) -> Linear(1280, 1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT, inplace=True),
            nn.Linear(in_features=in_features, out_features=config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC if possible (requires both classes to be present)
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
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / len(loader.dataset)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    train_metadata_path=config.TRAIN_METADATA_PATH,
    val_metadata_path=config.VAL_METADATA_PATH,
    epochs=config.EPOCHS,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    train_loader = data.get_dataloader(
        train_metadata_path, is_train=True, shuffle=True, batch_size=config.BATCH_SIZE
    )
    val_loader = data.get_dataloader(
        val_metadata_path, is_train=False, shuffle=False, batch_size=config.BATCH_SIZE
    )

    # Model Setup
    model = AsymmetricEfficientNet().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} [{duration:.1f}s] - "
            f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")


# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------


def inference(
    test_metadata_path=config.TEST_METADATA_PATH,
    model_path=config.MODEL_SAVE_PATH,
    submission_path=config.SUBMISSION_FILE,
):
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Metadata to get IDs in order
    if not os.path.exists(test_metadata_path):
        print("Test metadata not found. Skipping inference.")
        return

    test_df = pd.read_csv(test_metadata_path)
    test_ids = test_df["BraTS21ID"].values

    # Data Loader
    test_loader = data.get_dataloader(
        test_metadata_path,
        is_train=False,
        shuffle=False,  # Crucial to maintain order
        batch_size=config.BATCH_SIZE,
    )

    # Load Model
    model = AsymmetricEfficientNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(
            f"Warning: Model file {model_path} not found. Using random initialization."
        )

    model.eval()
    predictions = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # TTA: 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # TTA: 2. Horizontal Flip (dim 3)
            inputs_h = torch.flip(inputs, [3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # TTA: 3. Vertical Flip (dim 2)
            inputs_v = torch.flip(inputs, [2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average Predictions
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            predictions.extend(avg_prob.cpu().numpy().flatten())

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
