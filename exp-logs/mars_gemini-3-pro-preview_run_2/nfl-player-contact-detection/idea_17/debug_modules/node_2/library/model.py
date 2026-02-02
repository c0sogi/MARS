import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from library.config import Config
from library.utils import compute_mcc


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Implemented using BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                           If None, no alpha weighting is applied.
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCEWithLogitsLoss computes -log(pt)
        # We use reduction='none' to apply focal weights element-wise first
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # pt is the probability of the true class
        # pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Focal term: (1 - pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        loss = focal_term * bce_loss

        # Alpha weighting
        if self.alpha is not None:
            # alpha_t = alpha if target=1 else (1-alpha)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ResidualBlock(nn.Module):
    """
    Dense Residual Block: Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        return out


class MMWIN(nn.Module):
    """
    Multi-Modal Wide-Input Network (MM-WIN).
    Accepts a flattened feature vector and processes it through a stack of Residual Blocks.
    Outputs raw logits.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(MMWIN, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Output Head (Logits)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Training loop with Early Stopping and Validation.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Using Focal Loss to handle class imbalance
    # alpha=0.8 assigns higher weight to the minority class (approx 1:4 weighting relative to majority)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    best_mcc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # Ensure shape [B, 1]

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                logits = model(inputs)
                probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        # Calculate MCC using a default threshold of 0.5 for monitoring
        # (Threshold optimization happens after training)
        val_preds_bin = (val_preds > 0.5).astype(int)
        val_mcc = compute_mcc(val_targets, val_preds_bin)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.5f} | Val MCC: {val_mcc:.5f}"
        )

        # --- Early Stopping ---
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Val MCC: {best_mcc:.5f}")

    return model


def optimize_threshold(model, val_loader, device):
    """
    Finds the decision threshold that maximizes MCC on the validation set.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    best_threshold = 0.5
    best_mcc = -1.0

    # Search range: 0.1 to 0.9
    thresholds = np.arange(0.1, 0.91, 0.01)

    for thresh in thresholds:
        preds = (all_probs > thresh).astype(int)
        mcc = compute_mcc(all_targets, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    print(f"Optimized Threshold: {best_threshold:.3f} | MCC: {best_mcc:.5f}")
    return best_threshold


def predict(model, test_loader, device, threshold=0.5):
    """
    Generates binary predictions for the test set.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            # Test loader might return just inputs or inputs+ids.
            # Assuming inputs is the first element if tuple.
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            preds = (probs > threshold).astype(int)
            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions)
