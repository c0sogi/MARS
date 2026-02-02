import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from library.config import Config
from library.features import POSITIONS_VOCAB
from library.dataset import NFLContactDataset


# --- Loss Function ---
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Focal Loss for binary classification.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: binary labels (0 or 1)

        # Calculate probabilities
        p = torch.sigmoid(inputs)

        # Initialize loss tensor
        loss = torch.zeros_like(inputs)

        # Calculate term for Class 1 (Contact)
        # Loss = -alpha * (1 - p)^gamma * log(p)
        # Add epsilon for numerical stability inside log
        loss = torch.where(
            targets == 1,
            self.alpha * (1 - p) ** self.gamma * (-torch.log(p + 1e-8)),
            loss,
        )

        # Calculate term for Class 0 (No Contact)
        # Loss = -(1 - alpha) * p^gamma * log(1 - p)
        loss = torch.where(
            targets == 0,
            (1 - self.alpha) * p**self.gamma * (-torch.log(1 - p + 1e-8)),
            loss,
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# --- Model Components ---


class GatedResidualBlock(nn.Module):
    """
    Residual Block with Gated Linear Unit (GLU) mechanism.
    Output = (Signal * Sigmoid(Gate)) + Input
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.signal_fc = nn.Linear(dim, dim)
        self.gate_fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        # Gated Mechanism
        signal = self.signal_fc(x)
        gate = torch.sigmoid(self.gate_fc(x))
        out = signal * gate

        out = self.dropout(out)

        # Residual Connection + Normalization
        out = out + residual
        out = self.norm(out)
        return out


class KinematicStream(nn.Module):
    """
    Deep stream for processing kinematic data with entity embeddings.
    """

    def __init__(self):
        super().__init__()

        # --- Embeddings ---
        # Positions: 0..N-1 mapped from vocab. N is Unknown.
        self.pos_embedding = nn.Embedding(
            len(POSITIONS_VOCAB) + 1, Config.EMBEDDING_DIM
        )

        # Teams: 0 (Home), 1 (Away). We map -1 (Missing) to 2.
        self.team_embedding = nn.Embedding(3, Config.EMBEDDING_DIM)

        # --- Dimensions ---
        # Continuous features + 4 categorical embeddings (P1 Pos, P1 Team, P2 Pos, P2 Team)
        in_dim = Config.KINEMATIC_INPUT_DIM + 4 * Config.EMBEDDING_DIM

        # --- Backbone ---
        self.project = nn.Sequential(
            nn.Linear(in_dim, Config.KINEMATIC_HIDDEN_DIM),
            nn.BatchNorm1d(Config.KINEMATIC_HIDDEN_DIM),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(Config.KINEMATIC_HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.KINEMATIC_LAYERS)
            ]
        )

        self.head = nn.Linear(Config.KINEMATIC_HIDDEN_DIM, 1)

    def forward(self, x_kin, x_cat):
        # x_cat shape: [Batch, 4] -> [P1_Pos, P1_Team, P2_Pos, P2_Team]

        # 1. Embed Categorical Features
        p1_pos = self.pos_embedding(x_cat[:, 0])

        # Handle missing team (-1) by mapping to index 2
        t1 = x_cat[:, 1]
        t1 = torch.where(t1 < 0, torch.tensor(2, device=t1.device), t1)
        p1_team = self.team_embedding(t1)

        p2_pos = self.pos_embedding(x_cat[:, 2])

        t2 = x_cat[:, 3]
        t2 = torch.where(t2 < 0, torch.tensor(2, device=t2.device), t2)
        p2_team = self.team_embedding(t2)

        # 2. Concatenate Embeddings and Continuous Features
        emb = torch.cat([p1_pos, p1_team, p2_pos, p2_team], dim=1)
        x = torch.cat([x_kin, emb], dim=1)

        # 3. Forward Pass through Backbone
        x = self.project(x)
        for block in self.blocks:
            x = block(x)

        return self.head(x)


class VisualStream(nn.Module):
    """
    Shallow stream for visual correction features.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.VISUAL_INPUT_DIM, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, 1),
        )

    def forward(self, x):
        return self.net(x)


class EGRVNet(nn.Module):
    """
    Enhanced Gated Residual-Visual Network.
    Fuses Kinematic and Visual streams via learnable residual connection.
    """

    def __init__(self):
        super().__init__()
        self.kinematic = KinematicStream()
        self.visual = VisualStream()

        # Learnable scalar for visual stream contribution
        # Initialized to 0.0 to start with pure kinematics, allowing gradient to determine weight
        self.lambda_vis = nn.Parameter(torch.tensor(0.0))

    def forward(self, kin, vis, cat):
        k_logit = self.kinematic(kin, cat)
        v_logit = self.visual(vis)

        # Residual Fusion
        return k_logit + self.lambda_vis * v_logit


# --- Training & Inference Logic ---


def train_model():
    """
    Trains the EGRVNet model using Focal Loss and Early Stopping.
    Optimizes decision threshold on validation set.
    """
    device = torch.device(Config.DEVICE)
    model = EGRVNet().to(device)

    criterion = SigmoidFocalLoss(alpha=Config.ALPHA, gamma=Config.GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Load Data
    print("Initializing Datasets...")
    train_ds = NFLContactDataset(split="train")
    val_ds = NFLContactDataset(split="validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    best_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    best_thresh_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        # Use tqdm for progress tracking
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False)
        for batch in pbar:
            kin = batch["kinematic"].to(device)
            vis = batch["visual"].to(device)
            cat = batch["categorical"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(kin, vis, cat)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                kin = batch["kinematic"].to(device)
                vis = batch["visual"].to(device)
                cat = batch["categorical"].to(device)
                target = batch["target"].to(device).unsqueeze(1)

                logits = model(kin, vis, cat)
                loss = criterion(logits, target)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # --- Threshold Optimization (MCC) ---
        y_true = np.vstack(all_targets)
        y_prob = np.vstack(all_preds)

        best_epoch_mcc = -1.0
        best_thresh = 0.5

        # Grid search for threshold
        thresholds = np.linspace(0.1, 0.9, 33)
        for t in thresholds:
            y_pred = (y_prob > t).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_epoch_mcc:
                best_epoch_mcc = mcc
                best_thresh = t

        print(
            f"Epoch {epoch+1}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}, Val MCC {best_epoch_mcc:.6f} (Thresh {best_thresh:.2f})"
        )

        # --- Early Stopping ---
        if best_epoch_mcc > best_mcc:
            best_mcc = best_epoch_mcc
            torch.save(model.state_dict(), best_model_path)
            np.save(best_thresh_path, np.array([best_thresh]))
            patience_counter = 0
            print(f"  -> New Best Model Saved! MCC: {best_mcc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    return best_mcc


def predict_and_submit():
    """
    Generates predictions on the test set using the best trained model and threshold.
    Saves results to submission.csv.
    """
    device = torch.device(Config.DEVICE)
    model = EGRVNet().to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    thresh_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")

    if not os.path.exists(model_path):
        print("No model found at", model_path)
        return

    print("Loading model and generating predictions...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if os.path.exists(thresh_path):
        threshold = float(np.load(thresh_path))
    else:
        threshold = 0.5
    print(f"Using optimized threshold: {threshold:.4f}")

    test_ds = NFLContactDataset(split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    contact_ids = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            kin = batch["kinematic"].to(device)
            vis = batch["visual"].to(device)
            cat = batch["categorical"].to(device)
            c_ids = batch["contact_id"]

            logits = model(kin, vis, cat)
            probs = torch.sigmoid(logits)

            preds = (probs > threshold).int().cpu().numpy().flatten()

            contact_ids.extend(c_ids)
            predictions.extend(preds)

    df_sub = pd.DataFrame({"contact_id": contact_ids, "contact": predictions})

    df_sub.to_csv("submission.csv", index=False)
    print(f"Submission saved to submission.csv with {len(df_sub)} rows.")
