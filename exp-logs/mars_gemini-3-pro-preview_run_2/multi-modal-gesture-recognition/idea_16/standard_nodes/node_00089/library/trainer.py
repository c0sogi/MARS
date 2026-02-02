import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm  # Allowed, but we will suppress if needed or use simple prints
import sys

from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.dataset import GestureDataset
from library.model import GLT_CRCN
from library.loss import CombinedLoss
from library.postprocessing import apply_median_filter, decode_predictions


class Trainer:
    """
    Trainer class for the GLT-CRCN model.
    Handles training, validation, early stopping, and inference.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = GLT_CRCN().to(self.device)

        # Initialize Loss
        self.criterion = CombinedLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Checkpoint path
        self.checkpoint_dir = Config.WORKING_DIR
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in dataloader:
            # Move data to device
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(features, mask)

            # Compute Loss
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            count += features.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, dataloader):
        """
        Runs validation to compute loss and Levenshtein Error Rate.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        total_lev_dist = 0
        total_gt_gestures = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)
                seq_labels = batch["seq_labels"]  # List of numpy arrays or tensors

                # Forward pass
                outputs = self.model(features, mask)

                # Compute Loss
                loss = self.criterion(outputs, targets, mask)
                running_loss += loss.item() * features.size(0)
                count += features.size(0)

                # -------------------------------------------------------------
                # Metric Calculation
                # -------------------------------------------------------------
                # Use Stage 3 outputs for final prediction
                # Shape: (B, T, C)
                stage3_probs = outputs["stage3_cls"]

                # Get lengths to slice predictions correctly
                lengths = batch["lengths"]

                for i in range(features.size(0)):
                    length = lengths[i]
                    # Get probability sequence for this sample
                    probs = stage3_probs[i, :length, :].cpu().numpy()

                    # Argmax to get labels
                    pred_labels = np.argmax(probs, axis=1)

                    # Post-processing: Median Filter
                    smoothed_labels = apply_median_filter(pred_labels, kernel_size=15)

                    # Decode: Collapse and remove background
                    pred_sequence = decode_predictions(smoothed_labels)

                    # Ground Truth Sequence
                    gt_sequence = seq_labels[i]
                    if isinstance(gt_sequence, torch.Tensor):
                        gt_sequence = gt_sequence.tolist()
                    elif isinstance(gt_sequence, np.ndarray):
                        gt_sequence = gt_sequence.tolist()

                    # Compute Levenshtein Distance
                    dist = compute_levenshtein(pred_sequence, gt_sequence)

                    total_lev_dist += dist
                    total_gt_gestures += len(gt_sequence)

        avg_loss = running_loss / count if count > 0 else 0.0

        # Avoid division by zero
        if total_gt_gestures == 0:
            lev_score = 0.0
        else:
            lev_score = total_lev_dist / total_gt_gestures

        return avg_loss, lev_score

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    ):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val Levenshtein Score: {val_score}"
            )

            # Early Stopping based on Levenshtein Score (Lower is better)
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with score: {best_score}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
        # Load best model for future use
        self.load_checkpoint(self.checkpoint_path)

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        Returns a dictionary: {sample_id: predicted_labels_array}
        """
        self.model.eval()
        predictions = {}

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                sample_ids = batch["sample_ids"]
                lengths = batch["lengths"]

                outputs = self.model(features, mask)
                stage3_probs = outputs["stage3_cls"]

                for i, sample_id in enumerate(sample_ids):
                    length = lengths[i]
                    # Extract valid frames
                    probs = stage3_probs[i, :length, :].cpu().numpy()

                    # Convert to labels immediately to save memory/space in dict
                    # (Post-processing handles the smoothing later)
                    labels = np.argmax(probs, axis=1)

                    predictions[sample_id] = labels

        return predictions

    def load_checkpoint(self, path):
        """Loads model weights from path."""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"Checkpoint not found at {path}")


def run_training(debug=False, epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Helper function to instantiate datasets, dataloaders, and run the training pipeline.
    """
    set_seed(Config.SEED)

    # Instantiate Datasets
    train_dataset = GestureDataset(split="train", augment=True, debug=debug)
    val_dataset = GestureDataset(split="val", augment=False, debug=debug)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    trainer.fit(train_loader, val_loader, num_epochs=epochs)

    return trainer


def run_inference(trainer, debug=False, batch_size=Config.BATCH_SIZE):
    """
    Helper function to run inference on test set and return predictions.
    """
    test_dataset = GestureDataset(split="test", augment=False, debug=debug)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GestureDataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    predictions = trainer.predict(test_loader)
    return predictions
