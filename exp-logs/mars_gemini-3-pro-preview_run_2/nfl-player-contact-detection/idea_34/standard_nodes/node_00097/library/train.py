import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, FocalLoss, optimize_threshold
from library.data_processing import PIRVDataset, generate_contact_features
from library.model import PIRVNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The PIRV-Net model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): AdamW optimizer.
        criterion (nn.Module): Focal Loss function.
        device (torch.device): Compute device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for x_kin, x_vis, targets in dataloader:
        x_kin = x_kin.to(device)
        x_vis = x_vis.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (N, 1)

        optimizer.zero_grad()

        # Forward pass (returns logits)
        logits = model(x_kin, x_vis)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_kin.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PIRV-Net model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Focal Loss function.
        device (torch.device): Compute device.

    Returns:
        tuple: (average_loss, best_mcc, best_threshold)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_kin, x_vis, targets in dataloader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(x_kin, x_vis)
            loss = criterion(logits, targets)

            running_loss += loss.item() * x_kin.size(0)

            # Convert logits to probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all batches
    y_true = np.vstack(all_targets).flatten()
    y_pred_prob = np.vstack(all_probs).flatten()

    # Optimize threshold for MCC
    best_thresh, best_mcc = optimize_threshold(
        y_true, y_pred_prob, steps=Config.THRESHOLD_SEARCH_STEPS
    )

    return epoch_loss, best_mcc, best_thresh


class Trainer:
    """
    Manages the training lifecycle of the PIRV-Net model.
    """

    def __init__(self, device=None):
        set_seed(Config.SEED)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.best_threshold = 0.5

        # Paths
        self.train_meta = os.path.join(Config.METADATA_DIR, "train.csv")
        self.val_meta = os.path.join(Config.METADATA_DIR, "validation.csv")
        self.train_tracking = os.path.join(
            Config.INPUT_DIR, "train_player_tracking.csv"
        )
        self.train_helmets = os.path.join(
            Config.INPUT_DIR, "train_baseline_helmets.csv"
        )

        self.model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        self.thresh_save_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")

    def load_data(self, debug_sample=None):
        """
        Loads and processes training and validation data.
        """
        print("Loading and processing training data...")
        X_kin_train, X_vis_train, y_train = generate_contact_features(
            self.train_meta,
            self.train_tracking,
            self.train_helmets,
            mode="train",
            load_cached_data=True,
        )

        print("Loading and processing validation data...")
        X_kin_val, X_vis_val, y_val = generate_contact_features(
            self.val_meta,
            self.train_tracking,
            self.train_helmets,
            mode="val",
            load_cached_data=True,
        )

        # Debugging: Subsample if requested
        if debug_sample is not None:
            print(f"Subsampling data to {debug_sample} samples for debugging.")
            indices_train = np.random.choice(
                len(X_kin_train), min(len(X_kin_train), debug_sample), replace=False
            )
            X_kin_train, X_vis_train, y_train = (
                X_kin_train[indices_train],
                X_vis_train[indices_train],
                y_train[indices_train],
            )

            indices_val = np.random.choice(
                len(X_kin_val), min(len(X_kin_val), debug_sample), replace=False
            )
            X_kin_val, X_vis_val, y_val = (
                X_kin_val[indices_val],
                X_vis_val[indices_val],
                y_val[indices_val],
            )

        return (X_kin_train, X_vis_train, y_train), (X_kin_val, X_vis_val, y_val)

    def fit(
        self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug_sample=None
    ):
        """
        Main training loop with early stopping.
        """
        # 1. Prepare Data
        (X_kin_train, X_vis_train, y_train), (X_kin_val, X_vis_val, y_val) = (
            self.load_data(debug_sample)
        )

        train_dataset = PIRVDataset(X_kin_train, X_vis_train, y_train)
        val_dataset = PIRVDataset(X_kin_val, X_vis_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # 2. Initialize Model
        input_dim_kin = X_kin_train.shape[1]
        input_dim_vis = X_vis_train.shape[1]

        print(
            f"Initializing PIRV-Net with Kinematic Dim: {input_dim_kin}, Visual Dim: {input_dim_vis}"
        )
        self.model = PIRVNet(input_dim_kin, input_dim_vis).to(self.device)

        # 3. Setup Optimization
        criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 4. Training Loop
        best_val_mcc = -1.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on device {self.device}...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model, train_loader, optimizer, criterion, self.device
            )
            val_loss, val_mcc, val_thresh = validate(
                self.model, val_loader, criterion, self.device
            )

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MCC: {val_mcc} | "
                f"Best Thresh: {val_thresh}"
            )

            # Early Stopping Check
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                self.best_threshold = val_thresh
                patience_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), self.model_save_path)
                # Save best threshold
                np.save(self.thresh_save_path, np.array(self.best_threshold))
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best Val MCC: {best_val_mcc}"
                    )
                    break

        print(f"Training complete. Best Validation MCC: {best_val_mcc}")
        print(f"Best model saved to {self.model_save_path}")
        print(f"Best threshold saved to {self.thresh_save_path}")
