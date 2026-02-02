import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from library import config, utils, model, dataset, data_processing


class Trainer:
    """
    Manages the training, validation, and evaluation of the K-CAN model.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize Optimizer and Loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = utils.FocalLoss(
            alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA
        )

        # Tracking best performance
        self.best_val_mcc = -1.0
        self.patience_counter = 0
        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Unpack tuple inputs (sequence, center_features)
            sequence, center_features = inputs

            # Move data to device
            sequence = sequence.to(self.device)
            center_features = center_features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model((sequence, center_features))

            # Compute loss
            loss = self.criterion(logits, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Runs validation on the validation set.
        Computes average loss and finds the optimal threshold for MCC.
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                sequence, center_features = inputs
                sequence = sequence.to(self.device)
                center_features = center_features.to(self.device)
                targets = targets.to(self.device)

                logits = self.model((sequence, center_features))
                loss = self.criterion(logits, targets)

                total_loss += loss.item()

                # Store probabilities for MCC calculation
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # Concatenate all batches
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Optimize threshold for this epoch
        best_thresh, best_mcc = self.optimize_threshold(all_targets, all_probs)

        return avg_loss, best_mcc, best_thresh

    def optimize_threshold(self, y_true, y_probs):
        """
        Performs a grid search on validation probabilities to find the
        decision threshold that maximizes the Matthews Correlation Coefficient.
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            mcc = utils.compute_mcc(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def fit(self, epochs=config.EPOCHS, patience=config.PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_mcc, val_thresh = self.validate_epoch()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MCC: {val_mcc} | "
                f"Best Thresh: {val_thresh}"
            )

            # Early Stopping Check
            if val_mcc > self.best_val_mcc:
                self.best_val_mcc = val_mcc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with MCC: {val_mcc}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation MCC: {self.best_val_mcc}")
        return self.best_val_mcc


def run_training(debug=False, load_cached=True):
    """
    Orchestrates the data loading, model initialization, and training process.
    """
    utils.set_seed()
    device = config.get_device()

    # 1. Data Processing
    engineer = data_processing.FeatureEngineer()

    print("Preparing Training Data...")
    X_train, y_train, _ = engineer.process_dataset(
        split="train", load_cached_data=load_cached, debug=debug
    )

    print("Preparing Validation Data...")
    X_val, y_val, _ = engineer.process_dataset(
        split="validation", load_cached_data=load_cached, debug=debug
    )

    # 2. Dataset & Loader
    train_dataset = dataset.ContactSequenceDataset(X_train, y_train)
    val_dataset = dataset.ContactSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Initialization
    net = model.KCAN().to(device)

    # 4. Training
    trainer = Trainer(net, train_loader, val_loader, device)
    trainer.fit(epochs=config.EPOCHS, patience=config.PATIENCE)

    # 5. Final Threshold Optimization using Best Model
    print("Loading best model for final threshold optimization...")
    net.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    net.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            sequence, center_features = inputs
            sequence = sequence.to(device)
            center_features = center_features.to(device)

            logits = net((sequence, center_features))
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    best_thresh, best_mcc = trainer.optimize_threshold(all_targets, all_probs)

    print(f"Final Optimized Threshold: {best_thresh}")
    print(f"Final Validation MCC: {best_mcc}")

    # Save best threshold
    thresh_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")
    np.save(thresh_path, np.array([best_thresh]))
    print(f"Best threshold saved to {thresh_path}")
