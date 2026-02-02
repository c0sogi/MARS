import os
import time
import torch
import torch.optim as optim
import numpy as np
from library.config import (
    WORKING_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    SEED,
    BACKGROUND_CLASS_ID,
)
from library.utils import setup_logger, set_seed, calculate_levenshtein
from library.data_loader import get_dataloaders
from library.model import MSE_GCN
from library.losses import TotalLoss


class Trainer:
    """
    Trainer class for the MSE-GCN model.
    """

    def __init__(self):
        # Set reproducibility
        set_seed(SEED)

        # Setup logging
        self.logger = setup_logger("train.log")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize Model
        self.model = MSE_GCN().to(self.device)

        # Initialize Loss
        self.criterion = TotalLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Checkpoint path
        self.checkpoint_dir = os.path.join(WORKING_DIR, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def decode_predictions(self, probs, mask):
        """
        Decodes frame-wise probabilities into gesture sequences.
        Args:
            probs: (B, T, C) Tensor of probabilities
            mask: (B, T) Tensor mask
        Returns:
            List of lists containing gesture IDs.
        """
        # Get class indices: (B, T)
        preds = torch.argmax(probs, dim=2)

        # Convert to numpy
        preds_np = preds.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        decoded_sequences = []
        for i in range(preds_np.shape[0]):
            # Filter by mask
            valid_len = int(mask_np[i].sum())
            sequence = preds_np[i, :valid_len]

            # Collapse consecutive duplicates and remove background
            clean_seq = []
            prev = -1
            for token in sequence:
                if token != prev:
                    if token != BACKGROUND_CLASS_ID:
                        clean_seq.append(int(token))
                    prev = token
            decoded_sequences.append(clean_seq)

        return decoded_sequences

    def decode_targets(self, targets, mask):
        """
        Decodes frame-wise targets into gesture sequences.
        """
        targets_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        decoded_sequences = []
        for i in range(targets_np.shape[0]):
            valid_len = int(mask_np[i].sum())
            sequence = targets_np[i, :valid_len]

            clean_seq = []
            prev = -1
            for token in sequence:
                if token != prev:
                    if token != BACKGROUND_CLASS_ID:
                        clean_seq.append(int(token))
                    prev = token
            decoded_sequences.append(clean_seq)

        return decoded_sequences

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        metrics_accum = {}

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            features = batch["features"].to(self.device)
            cls_targets = batch["cls_labels"].to(self.device)
            bnd_targets = batch["bnd_labels"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            stage_outputs = self.model(features, mask, lengths)

            # Compute loss
            loss, metrics = self.criterion(
                stage_outputs, cls_targets, bnd_targets, mask
            )

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics
        avg_loss = epoch_loss / len(train_loader)
        for k in metrics_accum:
            metrics_accum[k] /= len(train_loader)

        duration = time.time() - start_time

        self.logger.info(
            f"Epoch {epoch} [Train] Loss: {avg_loss} Time: {duration:.2f}s"
        )
        # Log detailed loss components
        details = " | ".join([f"{k}: {v}" for k, v in metrics_accum.items()])
        self.logger.info(f"Epoch {epoch} [Train Details] {details}")

        return avg_loss

    def validate(self, val_loader, epoch):
        self.model.eval()
        epoch_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                cls_targets = batch["cls_labels"].to(self.device)
                bnd_targets = batch["bnd_labels"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                # Forward pass
                stage_outputs = self.model(features, mask, lengths)

                # Compute loss
                loss, _ = self.criterion(stage_outputs, cls_targets, bnd_targets, mask)
                epoch_loss += loss.item()

                # Decode predictions from Stage 3 (Final Stage)
                final_stage_out = stage_outputs[-1]
                cls_probs = final_stage_out["cls"]

                batch_preds = self.decode_predictions(cls_probs, mask)
                batch_targets = self.decode_targets(cls_targets, mask)

                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets)

        avg_loss = epoch_loss / len(val_loader)
        levenshtein_score = calculate_levenshtein(all_preds, all_targets)

        self.logger.info(
            f"Epoch {epoch} [Val] Loss: {avg_loss} Levenshtein: {levenshtein_score}"
        )

        return avg_loss, levenshtein_score

    def fit(self):
        self.logger.info("Starting training process...")

        # Get dataloaders
        train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE, load_cached_data=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            # Train
            self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_score = self.validate(val_loader, epoch)

            # Checkpointing and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(
                    f"New best model saved at epoch {epoch} with Val Loss: {val_loss}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{PATIENCE}"
                )

            if patience_counter >= PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss}")
