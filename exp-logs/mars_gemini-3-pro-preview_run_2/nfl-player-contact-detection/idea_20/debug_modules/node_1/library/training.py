import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
from library import config, dataset, models

# Set seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: binary labels (0 or 1)

        # BCE with logits (no reduction yet)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Probabilities
        pt = torch.exp(-bce_loss)  # p_t is the probability of the true class

        # Alpha weighting
        # If target=1, weight = alpha. If target=0, weight = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Hyperparameters
        self.epochs = config.TRAIN_PARAMS["epochs"]
        self.patience = config.TRAIN_PARAMS["early_stopping_patience"]
        self.lr = config.TRAIN_PARAMS["learning_rate"]
        self.weight_decay = config.TRAIN_PARAMS["weight_decay"]

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss Function
        self.criterion = FocalLoss(
            alpha=config.TRAIN_PARAMS["focal_alpha"],
            gamma=config.TRAIN_PARAMS["focal_gamma"],
        )

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x_kin, x_vis, y) in enumerate(self.train_loader):
            x_kin = x_kin.to(self.device)
            x_vis = x_vis.to(self.device)
            y = y.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            logits = self.model(x_kin, x_vis)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_kin, x_vis, y in self.val_loader:
                x_kin = x_kin.to(self.device)
                x_vis = x_vis.to(self.device)
                y = y.to(self.device).view(-1, 1)

                logits = self.model(x_kin, x_vis)
                loss = self.criterion(logits, y)

                running_loss += loss.item()

                # Store probabilities and targets for MCC calculation
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()

        # Find best MCC by searching thresholds
        best_mcc = -1.0
        best_threshold = 0.5

        # Search range: 0.1 to 0.9
        thresholds = np.arange(0.1, 0.95, 0.05)
        for t in thresholds:
            bin_preds = (all_preds >= t).astype(int)
            mcc = matthews_corrcoef(all_targets, bin_preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = t

        return avg_loss, best_mcc, best_threshold

    def fit(self):
        best_mcc = -float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_mcc, val_thresh = self.validate()

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {val_mcc:.6f} (Thresh: {val_thresh:.2f})"
            )

            # Early Stopping Check based on MCC
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved! MCC: {best_mcc:.6f}")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val MCC: {best_mcc:.6f}")


def train_model():
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader = dataset.get_train_val_loaders()

    # 3. Model Initialization
    print("Initializing GRV-Net Model...")
    model = models.GRVNet().to(device)

    # 4. Trainer Initialization
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Execute Training
    trainer.fit()
