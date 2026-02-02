import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import CascadedNet
from library.losses import CombinedLoss

# Set fixed seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class Trainer:
    """
    Manages the training and validation lifecycle of the Cascaded Refinement Network.
    """

    def __init__(self, train_loader, val_loader, device=None):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data (windowed).
            val_loader (DataLoader): DataLoader for validation data (full sequences).
            device (torch.device, optional): Device to run on. Defaults to auto-detect.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Trainer initialized on device: {self.device}")

        # Initialize Model
        self.model = CascadedNet().to(self.device)

        # Initialize Loss
        self.criterion = CombinedLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        Returns:
            dict: Average losses for the epoch.
        """
        self.model.train()

        running_loss = 0.0
        running_ce1 = 0.0
        running_ce2 = 0.0
        running_smooth = 0.0
        total_batches = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # features: (Batch, Time, Input_Dim)
            # labels: (Batch, Time)
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # s1_logits: (Batch, Time, Num_Classes)
            # s2_logits: (Batch, Time, Num_Classes)
            s1_logits, s2_logits = self.model(features)

            # Compute Loss
            loss_dict = self.criterion(s1_logits, s2_logits, labels)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            running_ce1 += loss_dict["ce1"].item()
            running_ce2 += loss_dict["ce2"].item()
            running_smooth += loss_dict["smooth"].item()
            total_batches += 1

        avg_metrics = {
            "loss": running_loss / total_batches if total_batches > 0 else 0.0,
            "ce1": running_ce1 / total_batches if total_batches > 0 else 0.0,
            "ce2": running_ce2 / total_batches if total_batches > 0 else 0.0,
            "smooth": running_smooth / total_batches if total_batches > 0 else 0.0,
        }

        return avg_metrics

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            dict: Average losses and accuracy.
        """
        self.model.eval()

        running_loss = 0.0
        correct_preds = 0
        total_frames = 0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Unpack based on dataset definition (features, labels, sample_id)
                # Validation loader returns full sequences, batch_size=1 usually
                features, labels, _ = batch_data

                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                s1_logits, s2_logits = self.model(features)

                # Compute Loss
                loss_dict = self.criterion(s1_logits, s2_logits, labels)
                running_loss += loss_dict["loss"].item()

                # Compute Accuracy using Stage 2 logits (Refined output)
                # s2_logits: (Batch, Time, Num_Classes)
                preds = torch.argmax(s2_logits, dim=2)

                # Mask out padding if necessary (though val loader usually yields valid lengths)
                # Here we assume batch_size=1 and valid full sequences
                correct_preds += (preds == labels).sum().item()
                total_frames += labels.numel()
                total_batches += 1

        metrics = {
            "val_loss": running_loss / total_batches if total_batches > 0 else 0.0,
            "val_accuracy": correct_preds / total_frames if total_frames > 0 else 0.0,
        }

        return metrics

    def fit(
        self, num_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE
    ):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {num_epochs} epochs...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.evaluate()

            # Print metrics (Full precision)
            print(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']}, "
                f"Train CE1: {train_metrics['ce1']}, "
                f"Train CE2: {train_metrics['ce2']}, "
                f"Train Smooth: {train_metrics['smooth']}, "
                f"Val Loss: {val_metrics['val_loss']}, "
                f"Val Accuracy: {val_metrics['val_accuracy']}"
            )

            # Checkpoint & Early Stopping
            current_val_loss = val_metrics["val_loss"]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")
