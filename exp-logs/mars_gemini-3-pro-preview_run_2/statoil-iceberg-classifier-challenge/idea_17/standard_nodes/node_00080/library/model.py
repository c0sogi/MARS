import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.data_loader import process_and_cache_data, IcebergDataset

# ==========================================
# 1. CUSTOM MODULES
# ==========================================


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Uses Max and Average pooling to refine features based on signal salience.
    """

    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # --- Channel Attention ---
        reduced_channels = max(channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()

        # --- Spatial Attention ---
        # Input channels = 2 (Avg, Max)
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # --- Channel Attention ---
        avg_pool = F.avg_pool2d(x, (h, w)).view(b, c)
        max_pool = F.max_pool2d(x, (h, w)).view(b, c)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x_channel = x * channel_att

        # --- Spatial Attention ---
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)

        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_concat))

        return x_channel * spatial_att


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Concatenates Max Pooling and Min Pooling outputs.
    This ensures that both peak signal intensity and shadow features are
    propagated to the next layer. Doubles the channel count.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Max Pool
        x_max = F.max_pool2d(x, self.kernel_size, self.stride)
        # Min Pool
        x_min = -F.max_pool2d(-x, self.kernel_size, self.stride)

        # Concatenate along channel dimension
        return torch.cat([x_max, x_min], dim=1)


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================


class WideDualPoolingNet(nn.Module):
    def __init__(self):
        super(WideDualPoolingNet, self).__init__()

        filters = Config.STAGE_FILTERS  # [64, 128, 128, 128]

        # --- Visual Branch ---

        # Block 1
        # Input: 3 channels -> 64 filters
        self.conv1 = nn.Conv2d(
            Config.NUM_CHANNELS, filters[0], kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.cbam1 = CBAM(filters[0])
        self.pool1 = DualPooling()  # Output: 64*2 = 128 channels

        # Block 2
        # Input: 128 channels -> 128 filters
        self.conv2 = nn.Conv2d(filters[0] * 2, filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam2 = CBAM(filters[1])
        self.pool2 = DualPooling()  # Output: 128*2 = 256 channels

        # Block 3
        # Input: 256 channels -> 128 filters
        self.conv3 = nn.Conv2d(filters[1] * 2, filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.relu3 = nn.ReLU(inplace=True)
        self.cbam3 = CBAM(filters[2])
        self.pool3 = DualPooling()  # Output: 128*2 = 256 channels

        # Block 4
        # Input: 256 channels -> 128 filters
        self.conv4 = nn.Conv2d(filters[2] * 2, filters[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.relu4 = nn.ReLU(inplace=True)
        self.cbam4 = CBAM(filters[3])
        self.pool4 = DualPooling()  # Output: 128*2 = 256 channels

        # --- Metadata Branch ---
        self.meta_fc = nn.Sequential(
            nn.Linear(1, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True)
        )

        # --- Fusion Head ---
        # Flattened visual size: 256 channels * 4 * 4 (spatial at end of block 4) = 4096
        self.fusion_dim = Config.LINEAR_INPUT_SIZE + 16

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Visual Branch
        x = self.pool1(self.cbam1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.pool2(self.cbam2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.pool3(self.cbam3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.pool4(self.cbam4(self.relu4(self.bn4(self.conv4(x)))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Metadata Branch
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)
        meta = self.meta_fc(angle)

        # Fusion
        fused = torch.cat([x, meta], dim=1)

        # Classification (Logits)
        out = self.classifier(fused)

        return out


# ==========================================
# 3. TRAINING & EVALUATION LOGIC
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Store predictions for metrics
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(targets)


def predict(model, loader, device):
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for images, angles, img_ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            ids.extend(img_ids)

    return np.array(ids), np.array(preds)


def run_training():
    print("Starting Shadow-Aware Wide-Body Network (SA-WBN) Training Pipeline...")

    # 1. Load Data
    data = process_and_cache_data(load_cached_data=True)

    # Combine Train and Val for Cross-Validation
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    ang_full = np.concatenate([data["ang_train"], data["ang_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    ids_full = np.concatenate([data["ids_train"], data["ids_val"]], axis=0)

    X_test = data["X_test"]
    ang_test = data["ang_test"]
    ids_test = data["ids_test"]

    # 2. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    test_preds_accum = np.zeros(len(ids_test))
    oof_preds = np.zeros(len(ids_full))
    oof_targets = np.zeros(len(ids_full))

    # Create Test Loader (Fixed)
    test_dataset = IcebergDataset(
        X_test, ang_test, labels=None, ids=ids_test, transform=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Prepare Fold Data
        train_ds = IcebergDataset(
            X_full[train_idx],
            ang_full[train_idx],
            y_full[train_idx],
            ids_full[train_idx],
            transform=True,
        )
        val_ds = IcebergDataset(
            X_full[val_idx],
            ang_full[val_idx],
            y_full[val_idx],
            ids_full[val_idx],
            transform=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = ShadowAwareWideBodyNet().to(Config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training Loop
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss, val_probs, val_targets = validate(
                model, val_loader, criterion, Config.DEVICE
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter == 0:
                print(
                    f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load Best Weights
        model.load_state_dict(best_model_wts)

        # Save Model
        model_path = os.path.join(Config.MODEL_DIR, f"sa_wbn_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        # OOF Predictions
        _, val_probs, val_targets_fold = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        # Map OOF preds back to original indices
        # Note: We need to handle index mapping carefully if we want full OOF array
        # For simplicity, we just calculate fold score here
        fold_score = log_loss(val_targets_fold, val_probs)
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} Log Loss: {fold_score:.6f}")

        # Test Inference (Ensemble)
        _, fold_test_preds = predict(model, test_loader, Config.DEVICE)
        test_preds_accum += fold_test_preds

        # Clean up
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # Average Test Predictions
    avg_test_preds = test_preds_accum / Config.NUM_FOLDS

    print("\n=== Cross-Validation Results ===")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.6f}")
    print(f"Average Log Loss: {np.mean(fold_scores):.6f}")

    # Save Submission
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

    # Ensure directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
