import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F

from library.config import Config
from library.data_processing import DataProcessor
from library.dataset import NFLDataset
from library.model import WIRKNet

# =============================================================================
# Loss Function
# =============================================================================


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha=Config.FOCAL_LOSS_ALPHA,
        gamma=Config.FOCAL_LOSS_GAMMA,
        reduction="mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [N, 1]
            targets: Binary targets [N, 1]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class

        # Alpha weighting
        # If target=1, alpha_t = alpha. If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Training & Evaluation Functions
# =============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (x_cat, x_cont, y) in enumerate(dataloader):
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x_cat, x_cont)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns avg loss, probabilities, and true labels.
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    return avg_loss, all_probs, all_targets


def optimize_threshold(y_true, y_probs):
    """
    Grid search for the best threshold maximizing MCC on validation data.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    thresholds = np.arange(
        Config.THRESHOLD_SEARCH_START,
        Config.THRESHOLD_SEARCH_END,
        Config.THRESHOLD_SEARCH_STEP,
    )

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


# =============================================================================
# Main Training Loop
# =============================================================================


def train_model():
    """
    Main driver for the WIRK-Net training pipeline.
    1. Loads data (cached or scratch).
    2. Initializes Model, Loss, Optimizer.
    3. Runs training loop with Early Stopping based on Validation MCC.
    4. Saves best model.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Preparation
    processor = DataProcessor()
    X_train, y_train, X_val, y_val = processor.get_train_val_data(load_cached_data=True)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    train_dataset = NFLDataset(X_train, y_train)
    val_dataset = NFLDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model Initialization
    # Determine input dimensions from dataset
    # NFLDataset automatically separates cat/cont features
    # We need to inspect one item or the internal attributes to get dimensions
    # Accessing internal attributes of the dataset for configuration
    num_cont_features = train_dataset.X_cont.shape[1]
    num_cat_features = train_dataset.X_cat.shape[1]

    # Determine vocab sizes for embeddings
    # We can infer this from the encoder in processor, but cleaner to check max index in data + 1
    # Note: If validation has UNKNOWN mapped to 0, and train has max index N, vocab size is N+1.
    # We should use the encoder classes count if available, or max from data.
    # processor.pos_encoder is available if we kept the instance, but here we just have X_train.
    # Let's compute max index per categorical column.
    if num_cat_features > 0:
        # Assuming all categorical columns are encoded with the same encoder (Position)
        # or we calculate per column.
        # In data_processing, we used one encoder for 'position'.
        # Let's calculate max index across the whole training set for each cat col.
        cat_vocab_sizes = []
        for i in range(num_cat_features):
            max_idx = train_dataset.X_cat[:, i].max().item()
            # Vocab size needs to handle 0..max_idx, so size is max_idx + 1
            # We add a buffer just in case, though safe_transform handles unknowns.
            cat_vocab_sizes.append(max_idx + 1)
    else:
        cat_vocab_sizes = []

    print(f"Model Configuration:")
    print(f"  Continuous Features: {num_cont_features}")
    print(f"  Categorical Features: {num_cat_features}")
    print(f"  Vocab Sizes: {cat_vocab_sizes}")

    model = WIRKNet(
        num_cont_features=num_cont_features,
        cat_vocab_sizes=cat_vocab_sizes,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RESIDUAL_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    criterion = FocalLoss(alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Training Loop
    best_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold & Metric
        curr_threshold, curr_mcc = optimize_threshold(val_targets, val_probs)

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCC: {curr_mcc} | "
            f"Best Thresh: {curr_threshold}"
        )

        # Early Stopping Check
        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_threshold
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # Save best threshold
            np.save(
                os.path.join(Config.WORKING_DIR, "best_threshold.npy"),
                np.array([best_threshold]),
            )
            print("  -> New best model saved!")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation MCC: {best_mcc}")
    print(f"Best Threshold: {best_threshold}")

    return model, best_threshold
