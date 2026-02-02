import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from library.utils import load_data, seed_everything

# ==========================================
# 1. Model Architecture Components
# ==========================================


class DualPooling(nn.Module):
    """
    Implements Dual-Stream Pooling: Concatenates Max Pooling (Peaks) and Min Pooling (Shadows).
    Preserves the full dynamic range of the radar signal.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # Min pooling is implemented as -Max(-X)
        self.min_pool_kernel = kernel_size
        self.min_pool_stride = stride

    def forward(self, x):
        max_p = self.max_pool(x)
        min_p = -F.max_pool2d(-x, self.min_pool_kernel, self.min_pool_stride)
        return torch.cat([max_p, min_p], dim=1)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) adapted for Radar.
    Uses Mixed Pooling (Max + Avg) for channel attention, avoiding Min pooling in attention logic.
    """

    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channels = channels

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.size()
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3))).view(b, c))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3))).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_out))
        x = x * spatial_att

        return x


class SWDINet(nn.Module):
    """
    Sustained-Width Delayed-Integration Network (SWDI-Net).
    Features:
    - Sustained Width: 128 filters in all conv stages.
    - Delayed Integration: 3x3 Conv integrates dual-pooled features (256ch) -> 128ch.
    - Strided Spatial-Integration Readout: Compresses 4x4x256 -> 2x2x128.
    """

    def __init__(self):
        super(SWDINet, self).__init__()

        # --- Visual Branch ---

        # Block 1
        # Input: 3 channels (HH, HV, Avg)
        self.block1_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block1_cbam = CBAMBlock(128)
        self.block1_pool = DualPooling(2, 2)  # Out: 128*2 = 256 ch

        # Block 2
        # Input: 256 channels from previous dual pool
        self.block2_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Integration
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block2_cbam = CBAMBlock(128)
        self.block2_pool = DualPooling(2, 2)  # Out: 256 ch

        # Block 3
        self.block3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_cbam = CBAMBlock(128)
        self.block3_pool = DualPooling(2, 2)  # Out: 256 ch

        # Block 4
        self.block4_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block4_cbam = CBAMBlock(128)
        self.block4_pool = DualPooling(2, 2)  # Out: 256 ch

        # Strided Spatial-Integration Readout
        # Input: 4x4 x 256 (Assuming 75x75 input -> 37 -> 18 -> 9 -> 4)
        self.readout = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        # Output: 2x2 x 128 -> Flatten -> 512

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )

        # --- Fusion Head ---
        # Visual (512) + Meta (32) = 544
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x_img, x_angle):
        # Visual Branch
        x = self.block1_conv(x_img)
        x = self.block1_cbam(x)
        x = self.block1_pool(x)

        x = self.block2_conv(x)
        x = self.block2_cbam(x)
        x = self.block2_pool(x)

        x = self.block3_conv(x)
        x = self.block3_cbam(x)
        x = self.block3_pool(x)

        x = self.block4_conv(x)
        x = self.block4_cbam(x)
        x = self.block4_pool(x)

        # Readout
        x = self.readout(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Metadata Branch
        # Handle NaN angles by replacing with mean (handled in preprocessing, but safe check)
        m = self.meta_mlp(x_angle)

        # Fusion
        combined = torch.cat([x, m], dim=1)
        out = self.fusion(combined)

        return out


# ==========================================
# 2. Training & Utility Functions
# ==========================================


def augment_batch(images):
    """
    Applies random rotations (0, 90, 180, 270) and horizontal flips.
    images: Tensor (B, C, H, W)
    """
    B, C, H, W = images.shape
    device = images.device

    # Random rotation k * 90 degrees
    k = torch.randint(0, 4, (B,), device=device)
    # Random horizontal flip
    flip = torch.rand(B, device=device) > 0.5

    out = images.clone()
    for i in range(B):
        img = out[i]
        if k[i] > 0:
            img = torch.rot90(img, k[i].item(), [1, 2])
        if flip[i]:
            img = torch.flip(img, [2])
        out[i] = img
    return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, angles, labels in loader:
        images, angles, labels = images.to(device), angles.to(device), labels.to(device)

        # Augmentation
        images = augment_batch(images)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, np.round(all_preds))
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images, angles, labels = (
                images.to(device),
                angles.to(device),
                labels.to(device),
            )
            outputs = model(images, angles)
            loss = criterion(outputs, labels.unsqueeze(1))

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, np.round(all_preds))
    return epoch_loss, epoch_acc, np.array(all_preds)


def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, angles in loader:
            images, angles = images.to(device), angles.to(device)
            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
    return np.array(all_preds)


def run_training_and_submission(
    epochs=50, batch_size=32, patience=10, seed=42, output_dir="./submission"
):
    """
    Main execution function.
    1. Loads data.
    2. Runs Stratified 5-Fold CV.
    3. Trains SWDI-Net with independent scaling per fold.
    4. Generates ensemble predictions.
    5. Saves submission file.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data = load_data(load_cached_data=True)
    X = data["X_train"]  # (N, 75, 75, 3)
    y = data["y_train"]
    inc_train = data["inc_angle_train"]
    ids_train = data["ids_train"]

    X_test_raw = data["X_test"]
    inc_test = data["inc_angle_test"]
    ids_test = data["ids_test"]

    # Fill missing incidence angles with global mean for now (refined per fold later if needed)
    # Note: Training data has no missing angles per description, but we check.
    # Test data has missing angles.
    global_inc_mean = np.nanmean(np.concatenate([inc_train, inc_test]))

    # Helper to fill NaNs
    def fill_inc(arr, fill_val):
        arr_filled = arr.copy()
        arr_filled[np.isnan(arr_filled)] = fill_val
        return arr_filled.reshape(-1, 1)

    inc_train_filled = fill_inc(inc_train, global_inc_mean)
    inc_test_filled = fill_inc(inc_test, global_inc_mean)

    # Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros((len(ids_test), 1))

    print(f"Starting 5-Fold Cross-Validation on {len(X)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== FOLD {fold} ===")

        # Split Data
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        inc_tr, inc_val = inc_train_filled[train_idx], inc_train_filled[val_idx]

        # Independent Per-Channel Min-Max Scaling
        # Calculate stats on training fold ONLY
        scalers = []
        X_tr_scaled = X_tr.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test_raw.copy()

        for c in range(3):
            c_min = X_tr[:, :, :, c].min()
            c_max = X_tr[:, :, :, c].max()
            denom = c_max - c_min + 1e-8

            X_tr_scaled[:, :, :, c] = (X_tr[:, :, :, c] - c_min) / denom
            X_val_scaled[:, :, :, c] = (X_val[:, :, :, c] - c_min) / denom
            X_test_scaled[:, :, :, c] = (X_test_raw[:, :, :, c] - c_min) / denom

        # Convert to Tensor
        # Permute to (N, C, H, W) for PyTorch
        def to_tensor(x_np, inc_np, y_np=None):
            x_t = torch.tensor(x_np.transpose(0, 3, 1, 2), dtype=torch.float32)
            inc_t = torch.tensor(inc_np, dtype=torch.float32)
            if y_np is not None:
                y_t = torch.tensor(y_np, dtype=torch.float32)
                return TensorDataset(x_t, inc_t, y_t)
            return TensorDataset(x_t, inc_t)

        train_ds = to_tensor(X_tr_scaled, inc_tr, y_tr)
        val_ds = to_tensor(X_val_scaled, inc_val, y_val)
        test_ds = to_tensor(X_test_scaled, inc_test_filled)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = SWDINet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, _ = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss:.6f} | Val Loss: {best_val_loss:.6f}"
                )

        # Load best weights
        model.load_state_dict(best_model_wts)

        # OOF Predictions
        _, _, val_probs = validate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_probs.flatten()

        # Test Predictions (Ensemble component)
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

        # Save model artifact
        torch.save(model.state_dict(), f"model_fold_{fold}.pth")

    # Average Test Predictions
    avg_test_preds = test_preds_accum / 5.0

    # Metrics
    total_log_loss = log_loss(y, oof_preds)
    total_acc = accuracy_score(y, np.round(oof_preds))
    print(f"\n=== CV Results ===")
    print(f"Overall Log Loss: {total_log_loss:.6f}")
    print(f"Overall Accuracy: {total_acc:.6f}")

    # Submission
    os.makedirs(output_dir, exist_ok=True)
    sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds.flatten()})

    # Ensure proper formatting
    sub_df["is_iceberg"] = sub_df["is_iceberg"].clip(
        0.001, 0.999
    )  # Clip for log loss safety
    sub_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
