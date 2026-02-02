import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from library.dataset import load_data, IcebergDataset, get_transforms, set_seed

# ==========================================
# Model Definitions
# ==========================================


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 64 -> 128 (Cite 00050: Early expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Angle Normalization (Cite 00054)
        self.bn_angle = nn.BatchNorm1d(1)

        # Classification Head
        # Global Max Pooling (128) + Angle (1) = 129
        self.fc1 = nn.Linear(129, 512)
        self.dropout = nn.Dropout(p=0.2)  # Reduced dropout (Cite 00017)
        self.fc2 = nn.Linear(512, 1)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x, angle):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global Max Pooling (Cite 00007, 00035)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)  # (N, 128)

        # Angle Normalization & Fusion (Cite 00054)
        angle = angle.view(-1, 1)
        angle = self.bn_angle(angle)

        features = torch.cat((x, angle), dim=1)  # (N, 129)

        # Head
        out = self.fc1(features)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).float().view(-1, 1)

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

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def run_training_pipeline():
    # Configuration
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    N_FOLDS = 5

    print(f"Starting SHSE-CNN Pipeline on {DEVICE}")

    # 1. Load Data
    print("Loading data...")
    X_train_raw, a_train_raw, y_train_raw, ids_train_raw = load_data("train")
    X_val_raw, a_val_raw, y_val_raw, ids_val_raw = load_data("val")

    # Merge for Cross-Validation
    X_all = np.concatenate([X_train_raw, X_val_raw], axis=0)
    a_all = np.concatenate([a_train_raw, a_val_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)

    # 2. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    best_model_paths = []
    cv_scores = []

    ckpt_dir = "./working/idea_15/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold} ---")

        # Prepare DataLoaders
        train_ds = IcebergDataset(
            X_all[train_idx],
            a_all[train_idx],
            y_all[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_all[val_idx],
            a_all[val_idx],
            y_all[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Initialize Model
        model = SHSE_CNN().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss = validate(model, val_loader, criterion, DEVICE)

            print(
                f"Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Save Best Model
        save_path = os.path.join(ckpt_dir, f"model_fold_{fold}.pth")
        torch.save(best_state, save_path)
        best_model_paths.append(save_path)
        cv_scores.append(best_loss)
        print(f"Fold {fold} Best Val Loss: {best_loss:.6f}")

    print(f"\nAverage CV Log Loss: {np.mean(cv_scores):.6f}")

    # 3. Inference and Submission
    print("\nGenerating Submission...")
    X_test, a_test, _, ids_test = load_data("test")
    test_ds = IcebergDataset(X_test, a_test, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    all_fold_preds = []

    for model_path in best_model_paths:
        model = SHSE_CNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        all_fold_preds.append(np.concatenate(fold_preds))

    # Average predictions across folds
    avg_preds = np.mean(all_fold_preds, axis=0).flatten()

    # Save to CSV
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")

    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# Execute the pipeline
run_training_pipeline()
