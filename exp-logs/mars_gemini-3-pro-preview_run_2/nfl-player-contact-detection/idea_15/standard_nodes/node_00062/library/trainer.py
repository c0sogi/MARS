import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.model import K_MLP
from library.data_processing import get_data_loaders


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for numerical stability with logits.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else Config.FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else Config.FOCAL_GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (N, 1)
        # targets: binary labels (N,)

        # Flatten inputs to match targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute standard BCE loss (returns -log(pt))
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Get probabilities
        pt = torch.exp(-bce_loss)

        # Calculate alpha weighting
        # alpha_t = alpha if target=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)

            # Convert logits to probabilities -> binary predictions (threshold 0.5 for monitoring)
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).float().cpu().numpy()
            targets = labels.view(-1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    val_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate MCC
    # Handle edge case where model predicts only one class
    if len(np.unique(all_preds)) < 2:
        val_mcc = 0.0
    else:
        val_mcc = matthews_corrcoef(all_targets, all_preds)

    return val_loss, val_mcc


def run_training():
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Data
    print("Initializing Data Loaders...")
    train_loader, val_loader, center_indices, scaler = get_data_loaders(
        load_cached_data=True
    )

    # Determine input dimension from a sample batch
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]

    print(f"Input Dimension: {input_dim}")
    print(f"Center Indices for Skip Connection: {center_indices}")

    # Initialize Model
    model = K_MLP(
        input_dim=input_dim,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Training Loop
    best_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCC: {val_mcc}"
        )

        # Early Stopping Check
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best MCC! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCC: {best_mcc}")
