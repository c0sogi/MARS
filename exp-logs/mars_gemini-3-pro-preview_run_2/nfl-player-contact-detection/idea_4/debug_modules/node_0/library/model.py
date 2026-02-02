import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
import library.config as config

# =============================================================================
# Model Architecture: Wide-Input Time-Resolved Gated Network (WI-TRGN)
# =============================================================================


class WITRGN(nn.Module):
    """
    Wide-Input Time-Resolved Gated Network.

    Architecture:
    1. Input: Wide feature vector (Batch, Window * Features).
    2. Reshape: (Batch, Features, Window).
    3. Encoder: 1D Convolutional Stack with Residuals (No Global Pooling).
    4. Heads: Dual Dense Heads (Player Interaction vs Ground Interaction).
    5. Gate: Hard gating based on 'is_ground' input.
    """

    def __init__(
        self,
        num_features=config.NUM_FEATURES_PER_STEP,
        window_size=config.WINDOW_SIZE,
        base_filters=64,
        dropout=0.3,
    ):
        super(WITRGN, self).__init__()

        self.num_features = num_features
        self.window_size = window_size

        # --- Temporal Encoder (1D CNN) ---
        # Layer 1: Expand features
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=base_filters,
            kernel_size=3,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(base_filters)

        # Layer 2: Residual Block
        self.conv2 = nn.Conv1d(
            in_channels=base_filters,
            out_channels=base_filters,
            kernel_size=3,
            padding="same",
        )
        self.bn2 = nn.BatchNorm1d(base_filters)

        # Layer 3: Deeper features
        self.conv3 = nn.Conv1d(
            in_channels=base_filters,
            out_channels=base_filters * 2,
            kernel_size=3,
            padding="same",
        )
        self.bn3 = nn.BatchNorm1d(base_filters * 2)

        # Flatten dimension: Channels * Window Size (No Global Pooling to preserve time context)
        self.flat_dim = (base_filters * 2) * window_size

        self.dropout = nn.Dropout(dropout)

        # --- Dual Heads ---
        # Head 1: Player-Player Interaction
        self.player_head = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Head 2: Player-Ground Interaction
        # Ground interactions often have distinct physics (sudden stop, vertical drop)
        self.ground_head = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, is_ground):
        """
        Args:
            x (Tensor): Wide features (Batch, Input_Dim)
            is_ground (Tensor): Binary indicator (Batch,) or (Batch, 1)
        """
        batch_size = x.size(0)

        # 1. Reshape Wide Input to Temporal Tensor
        # Input is constructed as [Block_t-5, Block_t-4, ..., Block_t+5]
        # Each block has 'num_features'.
        # View as (Batch, Window, Features) then transpose to (Batch, Features, Window)
        x = x.view(batch_size, self.window_size, self.num_features)
        x = x.transpose(1, 2)  # -> (Batch, Channels, Length)

        # 2. Convolutional Encoder
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual connection
        identity = x
        out = F.relu(self.bn2(self.conv2(x)))
        x = out + identity

        x = F.relu(self.bn3(self.conv3(x)))

        # 3. Flatten
        x = x.view(batch_size, -1)
        x = self.dropout(x)

        # 4. Dual Heads
        player_logits = self.player_head(x)
        ground_logits = self.ground_head(x)

        # 5. Gating
        # Ensure is_ground is (Batch, 1)
        if is_ground.dim() == 1:
            is_ground = is_ground.unsqueeze(1)

        # Hard Gate: Select ground logic if is_ground=1, else player logic
        final_logits = player_logits * (1 - is_ground) + ground_logits * is_ground

        return final_logits


# =============================================================================
# Training and Evaluation Logic
# =============================================================================


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for features, targets, is_ground in loader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)
        is_ground = is_ground.to(device)

        optimizer.zero_grad()

        logits = model(features, is_ground)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, is_ground in loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            is_ground = is_ground.to(device)

            logits = model(features, is_ground)
            loss = criterion(logits, targets)

            total_loss += loss.item() * features.size(0)

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate MCC
    preds_binary = (all_preds > threshold).astype(int)
    mcc = matthews_corrcoef(all_targets, preds_binary)

    return total_loss / len(loader.dataset), mcc, all_preds, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=config.EPOCHS,
    lr=config.LEARNING_RATE,
    patience=config.EARLY_STOPPING_PATIENCE,
    pos_weight=config.POS_WEIGHT,
    save_path=config.MODEL_SAVE_PATH,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Weighted BCE Loss to handle class imbalance (~1:72)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_mcc = -1.0
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(
        f"Config: Epochs={epochs}, LR={lr}, PosWeight={pos_weight}, Patience={patience}"
    )

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcc, _, _ = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MCC: {val_mcc:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! MCC: {best_mcc:.10f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCC: {best_mcc:.10f}")

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def optimize_threshold(model, val_loader):
    """
    Finds the optimal decision threshold on the validation set to maximize MCC.
    This is crucial because the weighted loss shifts probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get all probabilities
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, targets, is_ground in val_loader:
            features = features.to(device)
            is_ground = is_ground.to(device)

            logits = model(features, is_ground)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Grid search
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_thresh = 0.5
    best_mcc = -1.0

    for t in thresholds:
        preds = (all_probs > t).astype(int)
        mcc = matthews_corrcoef(all_targets, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    print(
        f"Threshold Optimization: Best Threshold = {best_thresh:.2f} (MCC: {best_mcc:.10f})"
    )
    return best_thresh


def generate_submission(
    model, test_loader, threshold, output_path=config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, _, is_ground in tqdm(test_loader, desc="Inference"):
            features = features.to(device)
            is_ground = is_ground.to(device)

            logits = model(features, is_ground)
            probs = torch.sigmoid(logits)

            # Apply threshold immediately to save memory/time
            preds = (probs > threshold).int().cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds).flatten()

    # Load sample submission to get IDs
    sub_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(all_preds) != len(sub_df):
        print(
            f"Warning: Prediction length ({len(all_preds)}) matches submission length ({len(sub_df)})."
        )
        # In case of mismatch, we might need to rely on the loader order being consistent with sample_submission
        # The metadata generation script ensured test.csv was derived from sample_submission,
        # so order should be preserved if loader shuffle=False.

    sub_df["contact"] = all_preds

    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
