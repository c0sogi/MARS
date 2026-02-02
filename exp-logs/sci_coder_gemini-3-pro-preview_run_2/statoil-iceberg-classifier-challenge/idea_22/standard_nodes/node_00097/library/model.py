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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Uses Mixed Pooling (Max + Avg) to handle noisy SAR data better than Max-only.
    Cite solution_lesson_node_00095.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


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
    Coordinate-Aware Dual-Pooling Network (Updated to DPCNet-like structure).
    Uses CBAM for mixed-pooling attention and reduced bottlenecks.
    """

    def __init__(self):
        super(CADPNet, self).__init__()
        filters = Config.FILTERS  # [64, 128, 128, 64]

        # --- Visual Branch ---
        # Stage 1: 3 -> 64 -> CBAM -> DualPool -> 128
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.cbam1 = CBAM(filters[0])
        self.pool1 = DualPooling()

        # Stage 2: 128 -> 128 -> CBAM -> DualPool -> 256
        # Input channels = filters[0] * 2 (due to DualPool)
        self.conv2 = nn.Conv2d(filters[0] * 2, filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.cbam2 = CBAM(filters[1])
        self.pool2 = DualPooling()

        # Stage 3: 256 -> 128 -> CBAM -> DualPool -> 256
        self.conv3 = nn.Conv2d(filters[1] * 2, filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.cbam3 = CBAM(filters[2])
        self.pool3 = DualPooling()

        # Stage 4: 256 -> 64 -> CBAM -> DualPool -> 128
        self.conv4 = nn.Conv2d(filters[2] * 2, filters[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.cbam4 = CBAM(filters[3])
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
        # Flattened visual: filters[3] * 2 (DualPool) * 4 * 4 spatial
        # With [64, 128, 128, 64], final is 128 * 16 = 2048
        final_channels = filters[3] * 2
        self.flat_dim = final_channels * 4 * 4

        self.fc1 = nn.Linear(self.flat_dim + 32, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fc2 = nn.Linear(512, 1)  # Logits

    def forward(self, x_img, x_angle):
        # Visual Branch
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.cbam1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        x_flat = x.view(x.size(0), -1)

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
