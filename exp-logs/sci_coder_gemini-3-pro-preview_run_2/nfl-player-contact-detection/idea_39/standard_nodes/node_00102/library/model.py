import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import compute_mcc

# =========================================================================
# Layers & Blocks
# =========================================================================


class InputBlock(nn.Module):
    """
    Applies fixed physical clamping and stochastic noise injection.
    """

    def __init__(self):
        super().__init__()
        self.clamp_min = Config.INPUT_CLAMP_MIN
        self.clamp_max = Config.INPUT_CLAMP_MAX
        self.noise_sigma = Config.NOISE_SIGMA

    def forward(self, x):
        # 1. Input Clamping (Physical Stability)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)

        # 2. Stochastic Noise Injection (Robustness)
        # Only apply during training
        if self.training and self.noise_sigma > 0:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise

        return x


class ResBlock(nn.Module):
    """
    Standard Residual Block with Dropout.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add(Skip) -> ReLU
    """

    def __init__(self, in_features, hidden_features=None, dropout=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)
        return out


class PyramidalBackbone(nn.Module):
    """
    Interleaved Pyramidal Backbone.
    Structure: Project -> ResBlock -> Project -> ResBlock ...
    """

    def __init__(self, input_dim, dims, dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for dim in dims:
            # Projection Layer
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())

            # Residual Block (Invariant processing at this scale)
            layers.append(ResBlock(dim, dropout=dropout))

            curr_dim = dim

        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x):
        return self.net(x)


# =========================================================================
# Main Model Architecture
# =========================================================================


class WDPIRVModel(nn.Module):
    """
    Wide-and-Deep Pyramidal Invariant Residual-Visual Network (WD-PIRV).
    Triple-Stream Network:
    1. Wide Stream: Linear Highway for dominant proximity signals.
    2. Deep Stream: Pyramidal Backbone for complex temporal dynamics.
    3. Visual Stream: Shallow MLP for visual correction.
    """

    def __init__(self, input_dim):
        super().__init__()

        # Input Preprocessing
        self.input_block = InputBlock()

        # Stream 1: Wide Kinematic Path (Linear Highway)
        self.wide = nn.Linear(input_dim, 1)

        # Stream 2: Deep Kinematic Path (Pyramidal Backbone)
        self.deep_backbone = PyramidalBackbone(
            input_dim, Config.PYRAMIDAL_DIMS, dropout=Config.DROPOUT_RATE
        )
        self.deep_head = nn.Linear(Config.PYRAMIDAL_DIMS[-1], 1)

        # Stream 3: Visual Path (Shallow Correction)
        # Note: We feed the full vector, relying on the shallow depth
        # to focus on the distinct visual features (or simple corrections).
        self.visual_net = nn.Sequential(
            nn.Linear(input_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, 1),
        )

        # Fusion Parameter (Learnable)
        self.visual_lambda = nn.Parameter(torch.tensor(Config.VISUAL_LAMBDA_INIT))

    def forward(self, x):
        # Preprocessing
        x = self.input_block(x)

        # Stream 1: Wide
        logit_wide = self.wide(x)

        # Stream 2: Deep
        deep_feat = self.deep_backbone(x)
        logit_deep = self.deep_head(deep_feat)

        # Stream 3: Visual
        logit_vis = self.visual_net(x)

        # Fusion
        # Logit_final = Logit_wide + Logit_deep + lambda * Logit_vis
        logit_final = logit_wide + logit_deep + self.visual_lambda * logit_vis

        return logit_final


# =========================================================================
# Loss Function
# =========================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probas = torch.sigmoid(logits)

        p_t = targets * probas + (1 - targets) * (1 - probas)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()


# =========================================================================
# Training & Inference Utilities
# =========================================================================


def train_model(model, train_loader, val_loader, device):
    """
    Handles the training loop with Early Stopping and Validation.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    best_mcc = -1.0
    patience_counter = 0
    best_threshold = 0.5

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds).flatten()
        val_targets = np.concatenate(val_targets).flatten()

        # Threshold Optimization
        thresholds = np.arange(0.1, 0.9, 0.05)
        mccs = []
        for t in thresholds:
            pred_binary = (val_preds > t).astype(int)
            mccs.append(matthews_corrcoef(val_targets, pred_binary))

        current_best_mcc = max(mccs)
        current_best_thresh = thresholds[np.argmax(mccs)]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCC: {current_best_mcc:.6f} (Thresh: {current_best_thresh:.2f})"
        )

        # --- Early Stopping ---
        if current_best_mcc > best_mcc:
            best_mcc = current_best_mcc
            best_threshold = current_best_thresh
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            np.save(Config.THRESHOLD_PATH, np.array([best_threshold]))
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best MCC: {best_mcc:.6f}")
    return best_threshold


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle cases where loader returns (features, labels) or just features
            if isinstance(batch, (list, tuple)):
                X_batch = batch[0]
            else:
                X_batch = batch

            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs).flatten()
