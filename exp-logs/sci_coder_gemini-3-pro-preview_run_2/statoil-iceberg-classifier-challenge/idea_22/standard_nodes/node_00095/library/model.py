import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import Config
from library.utils import EarlyStopping, seed_everything
from library.dataset import get_data, make_dataloaders, make_test_dataloader

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================


class MaxCoordinateAttention(nn.Module):
    """
    Coordinate Attention using 1D Max Pooling to preserve peak signals.
    Decomposes attention into vertical and horizontal directions.
    """

    def __init__(self, in_channels, reduction=16):
        super(MaxCoordinateAttention, self).__init__()
        # Adaptive Max Pool to get (H, 1) and (1, W)
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        # Ensure bottleneck has reasonable capacity
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Max Pooling along axes
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, 1, W) -> (N, C, W, 1)

        # Concatenate along spatial dimension for shared processing
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Shared 1x1 Conv reduction
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into H and W components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # Apply attention
        out = identity * a_h * a_w
        return out


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Concatenates MaxPool (peaks) and MinPool (shadows).
    Doubles the channel dimension.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Max Pooling
        x_max = F.max_pool2d(x, self.kernel_size, self.stride)
        # Min Pooling: -max(-x)
        x_min = -F.max_pool2d(-x, self.kernel_size, self.stride)
        # Concatenate along channel dimension
        return torch.cat([x_max, x_min], dim=1)


class CADPNet(nn.Module):
    """
    Coordinate-Aware Dual-Pooling Network.
    Integrates 4-stage CNN with channel contraction, Max-Coord Attention,
    Dual Pooling, and Metadata fusion.
    """

    def __init__(self):
        super(CADPNet, self).__init__()

        # --- Visual Branch ---
        # Filters: 64 -> 128 -> 128 -> 32

        # Stage 1: 3 -> 64 -> Attn -> Pool -> 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.ca1 = MaxCoordinateAttention(64)
        self.pool1 = DualPooling()

        # Stage 2: 128 -> 128 -> Attn -> Pool -> 256
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.ca2 = MaxCoordinateAttention(128)
        self.pool2 = DualPooling()

        # Stage 3: 256 -> 128 -> Attn -> Pool -> 256
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.ca3 = MaxCoordinateAttention(128)
        self.pool3 = DualPooling()

        # Stage 4: 256 -> 32 -> Attn -> Pool -> 64 (Contraction)
        self.conv4 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.ca4 = MaxCoordinateAttention(32)
        self.pool4 = DualPooling()

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # --- Fusion Head ---
        # Flattened visual: 64 channels * 4 * 4 spatial = 1024
        # Metadata: 32
        self.fc1 = nn.Linear(1024 + 32, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(512, 1)  # Logits

    def forward(self, x_img, x_angle):
        # Visual Branch
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.ca1(x)
        x = self.pool1(x)  # (B, 128, 37, 37)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.ca2(x)
        x = self.pool2(x)  # (B, 256, 18, 18)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.ca3(x)
        x = self.pool3(x)  # (B, 256, 9, 9)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.ca4(x)
        x = self.pool4(x)  # (B, 64, 4, 4)

        x_flat = x.view(x.size(0), -1)  # (B, 1024)

        # Metadata Branch
        angle = x_angle.view(-1, 1)
        x_meta = self.meta_mlp(angle)

        # Fusion
        x_concat = torch.cat([x_flat, x_meta], dim=1)

        x_out = self.fc1(x_concat)
        x_out = self.bn_fc1(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        logits = self.fc2(x_out)

        return logits


# ==========================================
# 2. TRAINING LOGIC
# ==========================================


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, angles, labels in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, labels in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def predict(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            angles = batch[1].to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds)


def run_training():
    """
    Main orchestration function.
    """
    Config.setup()
    seed_everything(Config.SEED)

    # Load Data
    train_data, test_data = get_data(load_cached_data=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Store fold models for ensemble
    fold_models = []

    # Cross Validation Loop
    for fold in range(Config.NUM_FOLDS):
        print(f"\nTraining Fold {fold + 1}/{Config.NUM_FOLDS}")

        # Prepare Data
        train_loader, val_loader, scaler_stats = make_dataloaders(
            train_data, fold_idx=fold
        )

        # Initialize Model
        model = CADPNet().to(device)

        # Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Early Stopping
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        early_stopping = EarlyStopping(
            patience=Config.PATIENCE, verbose=True, path=checkpoint_path
        )

        # Training Loop
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load best model for this fold
        model.load_state_dict(torch.load(checkpoint_path))
        fold_models.append((model, scaler_stats))

    # ==========================================
    # 3. SUBMISSION GENERATION
    # ==========================================
    print("\nGenerating submission...")

    # We need to predict using each fold's model and scaler stats
    test_ids = test_data["ids"]
    ensemble_preds = []

    for i, (model, stats) in enumerate(fold_models):
        print(f"Predicting with model from Fold {i+1}")
        # Recreate test loader with fold-specific scaling
        test_loader = make_test_dataloader(test_data, scaler_stats=stats)
        preds = predict(model, test_loader, device)
        ensemble_preds.append(preds)

    # Average predictions
    avg_preds = np.mean(ensemble_preds, axis=0).flatten()

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    with open(Config.SUBMISSION_PATH, "w") as f:
        f.write("id,is_iceberg\n")
        for pid, pred in zip(test_ids, avg_preds):
            f.write(f"{pid},{pred}\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
