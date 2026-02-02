import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from timm.layers import DropBlock2D

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import process_and_cache_data, IcebergDataset

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module with Global Average Pooling.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DRHACNN(nn.Module):
    """
    DropBlock-Regularized Hybrid-Attentive Plain CNN.
    """

    def __init__(self):
        super(DRHACNN, self).__init__()

        channels = Config.CHANNEL_CONFIG

        # Stage 1: Conv -> BN -> LeakyReLU -> SE -> MaxPool
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                Config.IN_CHANNELS, channels[0], kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(channels[0]),
            nn.MaxPool2d(2),
        )

        # Stage 2: Conv -> BN -> LeakyReLU -> SE -> MaxPool
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(channels[1]),
            nn.MaxPool2d(2),
        )

        # Stage 3: Conv -> BN -> LeakyReLU -> SE -> DropBlock -> MaxPool
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(channels[2]),
        )
        self.stage3_drop = DropBlock2D(
            drop_prob=0.0, block_size=Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage3_pool = nn.MaxPool2d(2)

        # Stage 4: Conv -> BN -> LeakyReLU -> SE -> DropBlock -> MaxPool
        self.stage4_conv = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(channels[3]),
        )
        self.stage4_drop = DropBlock2D(
            drop_prob=0.0, block_size=Config.DROPBLOCK_BLOCK_SIZE
        )
        self.stage4_pool = nn.MaxPool2d(2)

        # Readout: Global Max Pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # Input: Stage 3 pooled (128) + Stage 4 pooled (128) + Angle (1)
        input_dim = channels[2] + channels[3] + 1
        hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, Config.NUM_CLASSES),
        )

    def set_drop_prob(self, prob):
        """
        Updates the drop probability for DropBlock layers (Linear Schedule).
        """
        self.stage3_drop.drop_prob = prob
        self.stage4_drop.drop_prob = prob

    def forward(self, x, angle):
        # x: (B, 3, 75, 75)
        # angle: (B,)

        x = self.stage1(x)
        x = self.stage2(x)

        # Stage 3
        x3 = self.stage3_conv(x)
        x3 = self.stage3_drop(x3)
        x3_pooled_map = self.stage3_pool(x3)

        # Stage 4
        x4 = self.stage4_conv(x3_pooled_map)
        x4 = self.stage4_drop(x4)
        x4_pooled_map = self.stage4_pool(x4)

        # Selective Hierarchical Max Pooling
        p3 = self.global_max_pool(x3_pooled_map).flatten(1)
        p4 = self.global_max_pool(x4_pooled_map).flatten(1)

        # Fusion with raw incidence angle
        ang = angle.view(-1, 1)
        features = torch.cat([p3, p4, ang], dim=1)

        logits = self.classifier(features)
        return logits.squeeze(1)


# -----------------------------------------------------------------------------
# Training & Inference Pipeline
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()

    # Linear Schedule for DropBlock: 0 -> Config.DROPBLOCK_PROB
    current_prob = Config.DROPBLOCK_PROB * (epoch / total_epochs)
    model.set_drop_prob(current_prob)

    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

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
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            batch_ids = batch["id"]

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds.extend(probs)
            ids.extend(batch_ids)

    return np.array(ids), np.array(preds)


def run_training_pipeline():
    print(f"Initializing Experiment: {Config.EXP_ID}")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load and Preprocess Data
    data = process_and_cache_data(load_cached_data=True)

    # Combine Train and Val for Stratified K-Fold
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # 5-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    test_preds_accum = np.zeros(len(X_test))

    # Setup Test Loader (Fixed)
    test_dataset = IcebergDataset(X_test, angle_test, ids=ids_test, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\nFold {fold + 1}/{Config.N_FOLDS}")

        # Split Data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        ang_tr, ang_va = angle_full[train_idx], angle_full[val_idx]

        # Augmentation
        train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        train_ds = IcebergDataset(X_tr, ang_tr, y_tr, transform=train_tf)
        val_ds = IcebergDataset(X_va, ang_va, y_va, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Initialize Model, Optimizer, Criterion
        model = DRHACNN().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0

        # Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, Config.EPOCHS
            )
            val_loss = validate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Save Checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                    },
                    True,
                    Config.CHECKPOINT_DIR,
                    fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load Best Model for Inference
        best_path = os.path.join(Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict on Test Set
        _, fold_preds = predict(model, test_loader, device)
        test_preds_accum += fold_preds

    # Ensemble Predictions
    avg_preds = test_preds_accum / Config.N_FOLDS

    # Generate Submission
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute the pipeline
run_training_pipeline()
