import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import timm

from library.config import Config
from library.dataset import MGMTDataset
from library.utils import set_seed


class MGMTNet(nn.Module):
    """
    Multi-Modal 2.5D Stacked CNN for MGMT promoter methylation prediction.
    Uses a lightweight EfficientNet-B0 backbone adapted for multi-channel input.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=Config.IN_CHANNELS,
        num_classes=1,
    ):
        super(MGMTNet, self).__init__()
        # Initialize the backbone with custom input channels
        # timm handles the adaptation of the first conv layer automatically
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x):
        # Forward pass through the backbone
        # Output shape: (Batch_Size, num_classes) - raw logits
        return self.backbone(x)


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
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (N, 1) for BCE

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect predictions for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where only one class is present in the batch/epoch
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
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
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / len(loader.dataset)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def train_model():
    """
    Orchestrates the training process:
    1. Loads data
    2. Initializes model, criterion, optimizer
    3. Runs training loop with Early Stopping
    4. Saves best model
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Metadata
    train_df = pd.read_parquet(Config.TRAIN_METADATA)
    val_df = pd.read_parquet(Config.VAL_METADATA)

    if Config.DEBUG:
        print(f"Debug mode: using {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Prepare Datasets and Loaders
    train_dataset = MGMTDataset(train_df, split_name="train", load_cached_data=True)
    val_dataset = MGMTDataset(val_df, split_name="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = MGMTNet().to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}, Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}, Val AUC: {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")


def predict_and_submit():
    """
    Loads the best trained model, generates predictions for the test set,
    and saves the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Test Data
    test_df = pd.read_parquet(Config.TEST_METADATA)

    test_dataset = MGMTDataset(test_df, split_name="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = MGMTNet().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(
            f"Warning: No trained model found at {Config.MODEL_PATH}. Using random weights."
        )

    model.eval()

    # 3. Inference
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Get IDs from dataset
    ids = test_dataset.get_ids()

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Convert BraTS21ID to integer as per sample submission format
    submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
