import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.utils import load_data, seed_everything
from library.data import get_kfold_loaders, get_test_loader

# ==================================================================================
# 1. MODEL ARCHITECTURE
# ==================================================================================


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Module.
    Computes both Max Pooling (Peaks) and Min Pooling (Shadows)
    and concatenates them along the channel dimension.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Max Pooling
        x_max = F.max_pool2d(x, self.kernel_size, self.stride)

        # Min Pooling: -Max(-x)
        # This effectively captures the darkest regions (shadows) which are critical in SAR
        x_min = -F.max_pool2d(-x, self.kernel_size, self.stride)

        # Concatenate along channel dimension
        return torch.cat([x_max, x_min], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
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
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DPCNet(nn.Module):
    """
    Dual-Pooling Contracted Network.
    Features a 4-stage backbone with Dual Pooling and channel contraction,
    plus a metadata branch for incidence angle.
    """

    def __init__(self):
        super(DPCNet, self).__init__()

        # --- Visual Branch ---
        # Stage 1: 3 -> 64 -> Pool(Dual) -> 128 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        self.pool1 = DualPooling(2, 2)

        # Stage 2: 128 -> 64 -> Pool(Dual) -> 128 channels
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        self.pool2 = DualPooling(2, 2)

        # Stage 3: 128 -> 128 -> Pool(Dual) -> 256 channels
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.pool3 = DualPooling(2, 2)

        # Stage 4 (Contracting): 256 -> 64 -> Pool(Dual) -> 128 channels
        # Relaxed contraction to preserve features (Cite solution_lesson_node_00055)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.cbam4 = CBAM(64)
        self.pool4 = DualPooling(2, 2)

        # Flattened size: 128 channels * 4 * 4 spatial = 2048
        self.visual_dim = 2048

        # --- Metadata Branch ---
        self.meta_fc = nn.Sequential(nn.Linear(1, 16), nn.BatchNorm1d(16), nn.ReLU())
        self.meta_dim = 16

        # --- Fusion Head ---
        self.fusion_dim = self.visual_dim + self.meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased to 0.5 (Cite solution_lesson_node_00002)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased to 0.5
            nn.Linear(256, 1),
        )

    def forward(self, inputs):
        img, angle = inputs

        # Visual Branch
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.cbam1(x)
        x = self.pool1(x)  # 75 -> 37

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)  # 37 -> 18

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)  # 18 -> 9

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)  # 9 -> 4

        x = x.view(x.size(0), -1)  # Flatten

        # Metadata Branch
        m = self.meta_fc(angle)

        # Fusion
        combined = torch.cat([x, m], dim=1)
        out = self.classifier(combined)

        return out


# ==================================================================================
# 2. TRAINING LOGIC
# ==================================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        img, angle = inputs
        img = img.to(device)
        angle = angle.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model((img, angle))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    true_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            img, angle = inputs
            img = img.to(device)
            angle = angle.to(device)
            targets = targets.to(device)

            outputs = model((img, angle))
            loss = criterion(outputs, targets)

            running_loss += loss.item() * img.size(0)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(true_labels)


def run_training(data_dict, n_folds=5, epochs=60, batch_size=32, device="cuda"):
    loaders = get_kfold_loaders(data_dict, batch_size=batch_size, n_splits=n_folds)

    best_models = []

    for fold, (train_loader, val_loader) in enumerate(loaders):
        print(f"\nStarting Fold {fold+1}/{n_folds}")

        model = DPCNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        # Cite debug_lesson_2: Remove deprecated 'verbose' argument from PyTorch schedulers
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, _, _ = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(
                    f"  Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f} [Saved]"
                )
            else:
                patience_counter += 1
                print(
                    f"  Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}"
                )

            if patience_counter >= patience:
                print(
                    f"  Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}"
                )
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model for this fold
        save_path = f"./working/dpcnet_fold_{fold}.pth"
        torch.save(model.state_dict(), save_path)
        best_models.append(save_path)

    return best_models


# ==================================================================================
# 3. INFERENCE & SUBMISSION
# ==================================================================================


def generate_submission(data_dict, model_paths, batch_size=32, device="cuda"):
    test_loader = get_test_loader(data_dict, batch_size=batch_size)
    test_ids = data_dict["test_ids"]

    # Placeholder for ensemble predictions
    ensemble_preds = np.zeros((len(test_ids), 1))

    print(f"\nGenerating predictions using {len(model_paths)} models...")

    for path in model_paths:
        model = DPCNet().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                img, angle = inputs
                img = img.to(device)
                angle = angle.to(device)

                outputs = model((img, angle))
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy())

        ensemble_preds += np.array(fold_preds)

    # Average predictions
    avg_preds = ensemble_preds / len(model_paths)

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds.flatten()})

    # Save
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run_pipeline():
    """
    Main entry point to run the full training and inference pipeline.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data_dict = load_data(load_cached_data=True)

    # Train
    model_paths = run_training(
        data_dict, n_folds=5, epochs=60, batch_size=32, device=device
    )

    # Predict
    generate_submission(data_dict, model_paths, batch_size=32, device=device)
