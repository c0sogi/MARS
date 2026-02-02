import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from library import config, utils

# Ensure reproducibility
utils.seed_everything(config.SEED)


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for Volcano Seismic Spectrograms.
    Implements Instance Standardization per sample.
    """

    def __init__(self, spectrograms, targets=None):
        """
        Args:
            spectrograms (np.ndarray): Shape (N, 10, H, W)
            targets (np.ndarray, optional): Shape (N,) - Log-scaled targets expected for training.
        """
        self.spectrograms = spectrograms
        self.targets = targets

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        # Shape: (10, H, W)
        spec = self.spectrograms[idx].astype(np.float32)

        # Instance Standardization: (x - mean) / std per sample
        # We compute stats over the entire (10, H, W) volume to preserve relative amplitudes between sensors
        # while normalizing the global energy level of the segment.
        mean = np.mean(spec)
        std = np.std(spec)

        # Avoid division by zero
        if std < 1e-6:
            std = 1.0

        spec_norm = (spec - mean) / std

        # Convert to tensor
        x = torch.from_numpy(spec_norm)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            return x


class EfficientNet10Ch(nn.Module):
    """
    EfficientNet-B0 modified to accept 10-channel input (one per sensor).
    """

    def __init__(self):
        super(EfficientNet10Ch, self).__init__()

        # Load pre-trained EfficientNet B0
        # We use weights=None (random init) or default.
        # Since the input domain (spectrograms) is very different from ImageNet,
        # and we change the first layer, training from scratch or partial transfer is valid.
        # Given the task complexity, we'll load default weights but expect the first layer to be re-inited.
        base_model = models.efficientnet_b0(weights="DEFAULT")

        # Modify the first convolutional layer
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_conv = base_model.features[0][0]

        # Create new Conv2d with in_channels=10
        new_first_conv = nn.Conv2d(
            in_channels=config.CNN_CONFIG["in_channels"],
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias,
        )

        # Initialize the new layer
        # We can average the weights of the original RGB channels to initialize the 10 channels
        # or just use Kaiming initialization. Kaiming is safer for distinct modalities.
        nn.init.kaiming_normal_(
            new_first_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        # Replace the layer
        base_model.features[0][0] = new_first_conv

        # Modify the classifier head for Regression (1 output)
        # Original classifier[1] is Linear(in_features=1280, out_features=1000)
        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(in_features, 1)

        self.model = base_model

    def forward(self, x):
        return self.model(x)


def train_vision_model(train_specs, train_targets, val_specs, val_targets, fold_idx):
    """
    Trains the EfficientNet model for a single fold.

    Args:
        train_specs (np.ndarray): Training spectrograms.
        train_targets (np.ndarray): Training targets (raw scale).
        val_specs (np.ndarray): Validation spectrograms.
        val_targets (np.ndarray): Validation targets (raw scale).
        fold_idx (int): Current fold index.

    Returns:
        np.ndarray: OOF predictions for the validation set (raw scale).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Vision Model on {device} for Fold {fold_idx}")

    # Apply Log-Scaling to Targets if configured
    if config.CNN_CONFIG["use_log_target"]:
        y_train_proc = np.log1p(train_targets)
        y_val_proc = np.log1p(
            val_targets
        )  # Used for loss calculation if needed, but we eval on MAE
    else:
        y_train_proc = train_targets
        y_val_proc = val_targets

    # Create Datasets
    train_dataset = VolcanoDataset(train_specs, y_train_proc)
    val_dataset = VolcanoDataset(
        val_specs, y_val_proc
    )  # Dataset gets log targets for consistency

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CNN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.CNN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = EfficientNet10Ch()
    model.to(device)

    # Optimization
    criterion = nn.L1Loss()  # MAE Loss on Log-Space targets
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.CNN_CONFIG["learning_rate"],
        weight_decay=config.CNN_CONFIG["weight_decay"],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.CNN_CONFIG["epochs"], eta_min=1e-6
    )

    # Tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    early_stop_counter = 0

    for epoch in range(config.CNN_CONFIG["epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_preds_log = []

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds_log.extend(outputs.cpu().numpy().flatten())

        val_preds_log = np.array(val_preds_log)

        # Convert predictions back to original scale for metric calculation
        if config.CNN_CONFIG["use_log_target"]:
            val_preds_orig = np.expm1(val_preds_log)
        else:
            val_preds_orig = val_preds_log

        # Ensure non-negative
        val_preds_orig = np.maximum(0, val_preds_orig)

        # Calculate MAE
        epoch_mae = utils.mae_score(val_targets, val_preds_orig)

        # Step Scheduler
        scheduler.step()

        # Print metrics
        print(
            f"Epoch {epoch+1}/{config.CNN_CONFIG['epochs']} | Train Loss: {epoch_train_loss:.6f} | Val MAE: {epoch_mae}"
        )

        # Early Stopping Check
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            # Save checkpoint
            save_path = os.path.join(config.CNN_MODEL_DIR, f"cnn_fold_{fold_idx}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.CNN_CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} Best Val MAE: {best_mae}")

    # Return best predictions for OOF
    # We need to re-run inference with best model
    model.load_state_dict(best_model_wts)
    model.eval()

    final_preds_log = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            final_preds_log.extend(outputs.cpu().numpy().flatten())

    final_preds_log = np.array(final_preds_log)
    if config.CNN_CONFIG["use_log_target"]:
        final_preds = np.expm1(final_preds_log)
    else:
        final_preds = final_preds_log

    return np.maximum(0, final_preds)


def predict_vision_model(test_specs, fold_idx):
    """
    Generates predictions for the test set using a trained model from a specific fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    test_dataset = VolcanoDataset(test_specs, targets=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.CNN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load Model
    model = EfficientNet10Ch()
    model_path = os.path.join(config.CNN_MODEL_DIR, f"cnn_fold_{fold_idx}.pth")

    if not os.path.exists(model_path):
        print(
            f"Warning: Model for fold {fold_idx} not found at {model_path}. Returning zeros."
        )
        return np.zeros(len(test_specs))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds_log = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_log.extend(outputs.cpu().numpy().flatten())

    preds_log = np.array(preds_log)

    # Inverse Transform
    if config.CNN_CONFIG["use_log_target"]:
        preds = np.expm1(preds_log)
    else:
        preds = preds_log

    return np.maximum(0, preds)
