import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from library.config import Config
from library.features import FeatureEngineer

# =============================================================================
# Custom Layers & Loss
# =============================================================================


class ClampingLayer(nn.Module):
    """
    Explicitly clamps input features to a bounded range to ensure numerical stability.
    """

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class TimeDistributedEncoder(nn.Module):
    """
    Applies a shared encoder to each time step independently.
    Input: (Batch, Time, Feats)
    Output: (Batch, Time, Hidden)
    """

    def __init__(self, input_dim_per_step, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_per_step, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, hidden_dim, dropout),
            ResidualBlock(hidden_dim, hidden_dim, dropout),
        )

    def forward(self, x):
        b, t, f = x.shape
        # Flatten time into batch dimension for shared processing
        x_flat = x.view(b * t, f)
        out_flat = self.encoder(x_flat)
        # Reshape back
        return out_flat.view(b, t, -1)


# =============================================================================
# Main Architecture: TD-SRN
# =============================================================================


class TD_SRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = Config.WINDOW_SIZE
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout = Config.DROPOUT_RATE
        self.visual_lambda = Config.VISUAL_LAMBDA

        # --- Feature Dimensions ---
        # Base features (9) * 2 players + distance (1) = 19 features per step
        self.kin_feats_per_step = 19

        # Visual features: 4 per player * 2 = 8
        self.vis_input_dim = 8

        # --- Kinematic Stream ---
        self.clamp = ClampingLayer(Config.CLAMP_MIN, Config.CLAMP_MAX)

        self.td_encoder = TimeDistributedEncoder(
            input_dim_per_step=self.kin_feats_per_step,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Aggregator takes flattened time-distributed features
        self.flattened_dim = self.window_size * self.hidden_dim

        self.kinematic_aggregator = nn.Sequential(
            nn.Linear(self.flattened_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),  # Kinematic Logit
        )

        # --- Visual Stream ---
        self.visual_mlp = nn.Sequential(
            nn.Linear(self.vis_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Visual Logit
        )

    def forward(self, x_kin, x_vis):
        """
        x_kin: (Batch, Window * FeatsPerStep) - Flattened window features
        x_vis: (Batch, VisFeats)
        """
        b = x_kin.shape[0]

        # 1. Kinematic Stream
        # Reshape to (Batch, Time, Feats)
        x_kin_reshaped = x_kin.view(b, self.window_size, self.kin_feats_per_step)

        # Clamp inputs for stability
        x_kin_clamped = self.clamp(x_kin_reshaped)

        # Shared Encoding
        x_encoded = self.td_encoder(x_kin_clamped)  # (Batch, Time, Hidden)

        # Flatten preserving temporal order
        x_flat = x_encoded.view(b, -1)

        # Aggregate
        kin_logit = self.kinematic_aggregator(x_flat)

        # 2. Visual Stream
        vis_logit = self.visual_mlp(x_vis)

        # 3. Residual Fusion
        final_logit = kin_logit + self.visual_lambda * vis_logit

        return final_logit


# =============================================================================
# Trainer & Data Handling
# =============================================================================


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.scaler_kin = StandardScaler()
        self.scaler_vis = StandardScaler()
        self.feature_engineer = FeatureEngineer()

    def _get_feature_columns(self):
        """
        Returns sorted lists of column names for kinematic and visual features
        to ensure correct tensor construction.
        """
        # Base kinematic features from features.py
        base_features = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "direction_sin",
            "direction_cos",
            "orientation_sin",
            "orientation_cos",
        ]

        kin_cols = []
        # Loop through lags in the exact order generated by features.py
        for lag in range(-Config.WINDOW_HALF, Config.WINDOW_HALF + 1):
            suffix = f"_t{lag:+d}" if lag != 0 else ""

            # P1 features
            for feat in base_features:
                kin_cols.append(f"p1_{feat}{suffix}")

            # P2 features
            for feat in base_features:
                kin_cols.append(f"p2_{feat}{suffix}")

            # Distance feature
            kin_cols.append(f"dist{suffix}")

        # Visual features (not windowed)
        vis_cols = []
        vis_base = ["left", "top", "width", "height"]
        for feat in vis_base:
            vis_cols.append(f"p1_vis_{feat}")
        for feat in vis_base:
            vis_cols.append(f"p2_vis_{feat}")

        return kin_cols, vis_cols

    def prepare_data(self, df, fit_scaler=False):
        kin_cols, vis_cols = self._get_feature_columns()

        X_kin = df[kin_cols].values.astype(np.float32)
        X_vis = df[vis_cols].values.astype(np.float32)

        if fit_scaler:
            X_kin = self.scaler_kin.fit_transform(X_kin)
            X_vis = self.scaler_vis.fit_transform(X_vis)
        else:
            X_kin = self.scaler_kin.transform(X_kin)
            X_vis = self.scaler_vis.transform(X_vis)

        return X_kin, X_vis

    def train(self):
        print("Generating/Loading Features...")
        df_train = self.feature_engineer.generate_features(split="train")
        df_val = self.feature_engineer.generate_features(split="validation")

        print("Preparing Tensors...")
        X_kin_train, X_vis_train = self.prepare_data(df_train, fit_scaler=True)
        y_train = df_train["contact"].values.astype(np.float32)

        X_kin_val, X_vis_val = self.prepare_data(df_val, fit_scaler=False)
        y_val = df_val["contact"].values.astype(np.float32)

        # Datasets
        train_dataset = TensorDataset(
            torch.tensor(X_kin_train),
            torch.tensor(X_vis_train),
            torch.tensor(y_train).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_kin_val),
            torch.tensor(X_vis_val),
            torch.tensor(y_val).unsqueeze(1),
        )

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

        # Model Setup
        model = TD_SRN().to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = FocalLoss(
            alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA
        )

        # Training Loop
        best_mcc = -1.0
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0

            for kin, vis, target in train_loader:
                kin, vis, target = (
                    kin.to(self.device),
                    vis.to(self.device),
                    target.to(self.device),
                )

                optimizer.zero_grad()
                logits = model(kin, vis)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * kin.size(0)

            train_loss /= len(train_dataset)

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for kin, vis, target in val_loader:
                    kin, vis, target = (
                        kin.to(self.device),
                        vis.to(self.device),
                        target.to(self.device),
                    )
                    logits = model(kin, vis)
                    loss = criterion(logits, target)
                    val_loss += loss.item() * kin.size(0)

                    probs = torch.sigmoid(logits)
                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(target.cpu().numpy())

            val_loss /= len(val_dataset)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)

            # Calculate MCC with default threshold 0.5 for monitoring
            # Note: Final threshold is optimized later
            pred_labels = (val_preds > 0.5).astype(int)
            current_mcc = matthews_corrcoef(val_targets, pred_labels)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MCC (0.5): {current_mcc:.10f}"
            )

            # Early Stopping & Model Checkpoint
            if current_mcc > best_mcc:
                best_mcc = current_mcc
                patience_counter = 0
                torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MCC: {best_mcc:.10f}")

        # Optimize Threshold
        print("Optimizing threshold...")
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=self.device)
        )
        model.eval()

        # Get all val preds again
        val_preds = []
        with torch.no_grad():
            for kin, vis, _ in val_loader:
                kin, vis = kin.to(self.device), vis.to(self.device)
                logits = model(kin, vis)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
        val_preds = np.concatenate(val_preds)

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_thresh = 0.5
        best_thresh_mcc = -1.0

        for t in thresholds:
            mcc = matthews_corrcoef(y_val, (val_preds > t).astype(int))
            if mcc > best_thresh_mcc:
                best_thresh_mcc = mcc
                best_thresh = t

        print(f"Best Threshold: {best_thresh} with MCC: {best_thresh_mcc:.10f}")
        self.best_threshold = best_thresh
        self.model = model

    def predict_and_submit(self):
        print("Generating Test Features...")
        df_test = self.feature_engineer.generate_features(split="test")

        print("Preparing Test Data...")
        X_kin_test, X_vis_test = self.prepare_data(df_test, fit_scaler=False)

        test_dataset = TensorDataset(torch.tensor(X_kin_test), torch.tensor(X_vis_test))

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for kin, vis in test_loader:
                kin, vis = kin.to(self.device), vis.to(self.device)
                logits = self.model(kin, vis)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        predictions = (all_preds > self.best_threshold).astype(int)

        # Create submission
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions.flatten()}
        )

        Config.setup_directories()
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    trainer = Trainer()
    trainer.train()
    trainer.predict_and_submit()
