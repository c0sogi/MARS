import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm

from library.config import Config
from library.utils import (
    set_seed,
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
)


# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------
class CadenceDataset(Dataset):
    """
    Dataset class for loading cadence snippets.
    Treats the (6, 273, 256) input as a 1-channel video sequence: (1, 6, 273, 256).
    """

    def __init__(self, metadata_df, mode="train"):
        self.metadata = metadata_df
        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load raw spectrogram: Shape (6, 273, 256)
        # Data is stored as float16, convert to float32 for training
        data = np.load(file_path).astype(np.float32)

        # 1. Normalization: Instance Standardization
        # Normalize each sample independently to handle varying noise floors
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-6:
            data = (data - mean) / std
        else:
            data = data - mean

        # 2. Augmentation (Train only)
        if self.mode == "train":
            # Random Horizontal Flip (Frequency axis = 256, axis 2)
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=2).copy()

            # Random Vertical Flip (Time axis = 273, axis 1)
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=1).copy()

            # Random Frequency Shift (Roll along frequency axis)
            # Simulates signals appearing at different frequencies
            if np.random.rand() > 0.5:
                shift = np.random.randint(-20, 20)
                data = np.roll(data, shift, axis=2)

        # 3. Format for R3D-18: (C, D, H, W) -> (1, 6, 273, 256)
        data = np.expand_dims(data, axis=0)
        tensor = torch.from_numpy(data)

        if self.mode == "test":
            return tensor, row["id"]
        else:
            return tensor, torch.tensor(row["target"], dtype=torch.float)


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
class CadenceModel2D(nn.Module):
    """
    2D CNN (ResNet34) with 6 input channels (one for each cadence frame).
    Cite Lesson 00010: Treating sequence steps as distinct spatial channels in a 2D CNN
    is more effective for alternating cadence patterns than 3D CNNs.
    """

    def __init__(self, pretrained=True):
        super(CadenceModel2D, self).__init__()

        # Use timm to create a ResNet34 with 6 input channels.
        # timm handles the weight initialization for the extra channels automatically.
        self.backbone = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            in_chans=6,
            num_classes=1,
            global_pool="avg",
        )

    def forward(self, x):
        # x shape: (Batch, 6, 273, 256)
        return self.backbone(x)


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Trains the SpatiotemporalResNet model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        print(f"Debug mode: Reducing dataset to {Config.DEBUG_SUBSET_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Create Datasets and Loaders
    train_dataset = CadenceDataset(train_df, mode="train")
    val_dataset = CadenceDataset(val_df, mode="val")

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

    # Initialize Model, Optimizer, Loss
    model = SpatiotemporalResNet(pretrained=True).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # Training Loop
    best_score = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = AverageMeter()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.update(loss.item(), inputs.size(0))

        # Validation
        model.eval()
        val_loss = AverageMeter()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss.update(loss.item(), inputs.size(0))

                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate Metric
        auc_score = calculate_roc_auc(all_targets, all_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.avg:.6f} | Val Loss: {val_loss.avg:.6f} | Val AUC: {auc_score:.15f}"
        )

        # Checkpoint & Early Stopping
        if auc_score > best_score:
            best_score = auc_score
            save_checkpoint(model, optimizer, scheduler, epoch, best_score)
            print(f"  New best model saved with AUC: {best_score:.15f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_score


# -----------------------------------------------------------------------------
# Inference Function
# -----------------------------------------------------------------------------
def predict_submission():
    """
    Loads the best model and generates predictions for the test set.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print("Starting inference...")

    # Load Test Data
    test_df = pd.read_csv(Config.TEST_METADATA)
    test_dataset = CadenceDataset(test_df, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = SpatiotemporalResNet(pretrained=False).to(device)
    checkpoint = load_checkpoint(model, device=device)

    if checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} (AUC: {checkpoint['score']:.5f})"
        )
    else:
        print("Warning: No checkpoint found. Using random weights.")

    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for inputs, batch_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids.extend(batch_ids)
            preds.extend(probs)

    # Save Submission
    submission = pd.DataFrame({"id": ids, "target": preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
