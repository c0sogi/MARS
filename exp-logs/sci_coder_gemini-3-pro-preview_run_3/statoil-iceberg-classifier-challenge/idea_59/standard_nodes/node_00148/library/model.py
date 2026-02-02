import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, get_logger

# Logger setup
logger = get_logger("model")

# ==========================================
# Data Processing & Caching
# ==========================================


def process_images(df):
    # Band 1 and Band 2 are lists of floats
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Synthetic 3rd channel (Average)
    b3 = (b1 + b2) / 2.0

    # Stack: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)
    return X


def get_data(load_cached_data=True):
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(Config.WORK_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORK_DIR, "y_train.npy"),
        "angles_train": os.path.join(Config.WORK_DIR, "angles_train.npy"),
        "ids_train": os.path.join(Config.WORK_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.WORK_DIR, "X_test.npy"),
        "angles_test": os.path.join(Config.WORK_DIR, "angles_test.npy"),
        "ids_test": os.path.join(Config.WORK_DIR, "ids_test.npy"),
    }

    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        logger.info("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
        return data

    logger.info("Processing data from scratch...")

    train_path = os.path.join(Config.INPUT_DIR, "train.json")
    test_path = os.path.join(Config.INPUT_DIR, "test.json")

    # Load Raw Data
    df_train = pd.read_json(train_path)
    df_test = pd.read_json(test_path)

    # Process Train
    X_train = process_images(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    angles_train = pd.to_numeric(df_train["inc_angle"], errors="coerce").values.astype(
        np.float32
    )
    ids_train = df_train["id"].values

    # Process Test
    X_test = process_images(df_test)
    angles_test = pd.to_numeric(df_test["inc_angle"], errors="coerce").values.astype(
        np.float32
    )
    ids_test = df_test["id"].values

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


# ==========================================
# Dataset
# ==========================================


class IcebergDataset(Dataset):
    def __init__(
        self,
        X,
        angles,
        y=None,
        transform=None,
        angle_scaler=None,
        angle_imputer_val=None,
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

        # Handle Angles
        angles_np = angles.reshape(-1, 1)
        if angle_imputer_val is not None:
            # Use provided value (for val/test)
            self.raw_angles = np.nan_to_num(angles_np, nan=angle_imputer_val).flatten()
        else:
            # Should not happen in this design as we pass processed angles, but safe fallback
            self.raw_angles = np.nan_to_num(angles_np, nan=0.0).flatten()

        # Normalization for AC-SE
        if angle_scaler is not None:
            self.norm_angles = angle_scaler.transform(
                self.raw_angles.reshape(-1, 1)
            ).flatten()
        else:
            self.norm_angles = self.raw_angles

        self.raw_angles = torch.FloatTensor(self.raw_angles)
        self.norm_angles = torch.FloatTensor(self.norm_angles)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        raw_ang = self.raw_angles[idx]
        norm_ang = self.norm_angles[idx]

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            return img, raw_ang, norm_ang, self.y[idx]
        else:
            return img, raw_ang, norm_ang


# ==========================================
# Model: ACI-CNN
# ==========================================


class AC_SE_Module(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Input: Channels + 1 (Normalized Angle)
        input_dim = channels + 1
        hidden_dim = max(channels // reduction, 8)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, angle):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # Concatenate normalized angle
        angle = angle.view(b, 1)
        y_cat = torch.cat([y, angle], dim=1)

        weights = self.fc(y_cat).view(b, c, 1, 1)
        return x * weights


class ACICNN(nn.Module):
    def __init__(self):
        super().__init__()

        def make_block(in_c, out_c):
            return nn.ModuleList(
                [
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
                    AC_SE_Module(out_c),
                    nn.MaxPool2d(2, 2),
                ]
            )

        # Stage 1: 3 -> 64
        self.s1_conv, self.s1_bn, self.s1_act, self.s1_se, self.s1_pool = make_block(
            3, 64
        )
        # Stage 2: 64 -> 128
        self.s2_conv, self.s2_bn, self.s2_act, self.s2_se, self.s2_pool = make_block(
            64, 128
        )
        # Stage 3: 128 -> 128
        self.s3_conv, self.s3_bn, self.s3_act, self.s3_se, self.s3_pool = make_block(
            128, 128
        )
        # Stage 4: 128 -> 128
        self.s4_conv, self.s4_bn, self.s4_act, self.s4_se, self.s4_pool = make_block(
            128, 128
        )

        # Readout Projections
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # Head
        # 64*2 (Stage3) + 64*2 (Stage4) = 256
        # + 1 (Raw Angle) = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_block(self, x, angle, conv, bn, act, se, pool):
        x = conv(x)
        x = bn(x)
        x = act(x)
        x = se(x, angle)
        x = pool(x)
        return x

    def forward(self, x, raw_angle, norm_angle):
        # Stage 1
        x1 = self.forward_block(
            x,
            norm_angle,
            self.s1_conv,
            self.s1_bn,
            self.s1_act,
            self.s1_se,
            self.s1_pool,
        )
        # Stage 2
        x2 = self.forward_block(
            x1,
            norm_angle,
            self.s2_conv,
            self.s2_bn,
            self.s2_act,
            self.s2_se,
            self.s2_pool,
        )
        # Stage 3
        x3 = self.forward_block(
            x2,
            norm_angle,
            self.s3_conv,
            self.s3_bn,
            self.s3_act,
            self.s3_se,
            self.s3_pool,
        )
        # Stage 4
        x4 = self.forward_block(
            x3,
            norm_angle,
            self.s4_conv,
            self.s4_bn,
            self.s4_act,
            self.s4_se,
            self.s4_pool,
        )

        # Readout Stage 3 (Isomorphic Dual-Polarity)
        p3 = self.proj3(x3)
        min_p3 = -F.adaptive_max_pool2d(-p3, 1)
        feat3 = torch.cat([F.adaptive_max_pool2d(p3, 1), min_p3], dim=1).flatten(1)

        # Readout Stage 4
        p4 = self.proj4(x4)
        min_p4 = -F.adaptive_max_pool2d(-p4, 1)
        feat4 = torch.cat([F.adaptive_max_pool2d(p4, 1), min_p4], dim=1).flatten(1)

        # Combine
        feats = torch.cat([feat3, feat4], dim=1)  # 256

        # Late Fusion with Raw Angle
        feats_aug = torch.cat([feats, raw_angle.view(-1, 1)], dim=1)  # 257

        out = self.head(feats_aug)
        return out


# ==========================================
# Training Logic
# ==========================================


def train_model():
    seed_everything(Config.SEED)

    # Load Data
    data = get_data()
    X = data["X_train"]
    y = data["y_train"]
    angles = data["angles_train"]

    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(X_test))

    # Augmentation
    class TrainTransform:
        def __call__(self, x):
            if np.random.rand() > 0.5:
                x = torch.flip(x, [2])  # Horizontal
            if np.random.rand() > 0.5:
                x = torch.flip(x, [1])  # Vertical
            return x

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"=== Fold {fold} ===")

        # Split Data
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        ang_tr, ang_val = angles[train_idx], angles[val_idx]

        # 1. Impute Angles (Median from Train only)
        tr_median = np.nanmedian(ang_tr)
        ang_tr_filled = np.nan_to_num(ang_tr, nan=tr_median)
        ang_val_filled = np.nan_to_num(ang_val, nan=tr_median)
        ang_test_filled = np.nan_to_num(angles_test, nan=tr_median)

        # 2. Fit Scaler on Train
        scaler = StandardScaler()
        scaler.fit(ang_tr_filled.reshape(-1, 1))

        train_ds = IcebergDataset(
            X_tr, ang_tr_filled, y_tr, transform=TrainTransform(), angle_scaler=scaler
        )
        val_ds = IcebergDataset(
            X_val, ang_val_filled, y_val, transform=None, angle_scaler=scaler
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model
        model = ACICNN().to(Config.DEVICE)
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0
            for imgs, raw_angs, norm_angs, labels in train_loader:
                imgs = imgs.to(Config.DEVICE)
                raw_angs = raw_angs.to(Config.DEVICE)
                norm_angs = norm_angs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                optimizer.zero_grad()
                outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss = 0
            preds = []
            with torch.no_grad():
                for imgs, raw_angs, norm_angs, labels in val_loader:
                    imgs = imgs.to(Config.DEVICE)
                    raw_angs = raw_angs.to(Config.DEVICE)
                    norm_angs = norm_angs.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE)

                    outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds.append(torch.sigmoid(outputs).cpu().numpy())

            val_loss /= len(val_ds)

            logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
                oof_preds[val_idx] = np.concatenate(preds)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        # Inference on Test
        model.load_state_dict(best_state)
        test_ds = IcebergDataset(
            X_test, ang_test_filled, y=None, transform=None, angle_scaler=scaler
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        fold_preds = []
        model.eval()
        with torch.no_grad():
            for imgs, raw_angs, norm_angs in test_loader:
                imgs = imgs.to(Config.DEVICE)
                raw_angs = raw_angs.to(Config.DEVICE)
                norm_angs = norm_angs.to(Config.DEVICE)
                outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
                fold_preds.append(torch.sigmoid(outputs).cpu().numpy())

        test_preds_accum += np.concatenate(fold_preds) / Config.NUM_FOLDS

    # Save Submission
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds_accum})
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")

    oof_score = log_loss(y, oof_preds)
    logger.info(f"Overall OOF Log Loss: {oof_score:.6f}")
