import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, get_logger, compute_metrics
from library.data import get_loaders, get_test_loader
from library.model import IcebergModel


class Trainer:
    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        self.logger = get_logger("trainer")

        # Initialize Model
        self.model = IcebergModel().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Checkpoint handling
        self.best_loss = float("inf")
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")

    def train_one_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        # Update DropBlock probability linearly
        # We can update per epoch or per batch. Per epoch is simpler and sufficient.
        # Linearly increase from start_prob to max_prob
        progress = epoch / total_epochs
        current_drop_prob = Config.DROPBLOCK_START_PROB + progress * (
            Config.DROPBLOCK_MAX_PROB - Config.DROPBLOCK_START_PROB
        )
        # Clamp to max
        current_drop_prob = min(current_drop_prob, Config.DROPBLOCK_MAX_PROB)

        self.model.set_dropblock_prob(current_drop_prob)

        start_time = time.time()

        for batch_idx, (images, angles, targets) in enumerate(train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images, angles)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Store predictions for metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)

        return epoch_loss, metrics, current_drop_prob

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images, angles)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)

        return epoch_loss, metrics

    def fit(self):
        set_seed(Config.SEED)

        # Get DataLoaders
        train_loader, val_loader = get_loaders(debug=self.debug)

        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(
            f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
        )

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss, train_metrics, drop_prob = self.train_one_epoch(
                train_loader, epoch, Config.EPOCHS
            )

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            self.logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"DropProb: {drop_prob:.4f} | "
                f"Train Loss: {train_loss:.6f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.6f} Acc: {val_metrics['accuracy']:.4f}"
            )

            # Checkpointing & Early Stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(
                    f"New best model saved with loss: {self.best_loss:.6f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        self.logger.info(f"Training finished. Best Val Loss: {self.best_loss:.6f}")

    def predict(self):
        self.logger.info("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            self.logger.error("No best model found to load for prediction.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        test_loader = get_test_loader(debug=self.debug)

        predictions = []
        ids = []

        with torch.no_grad():
            for images, angles, img_ids in test_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)

                logits = self.model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy()

                predictions.extend(probs)
                ids.extend(img_ids)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": ids, "is_iceberg": predictions})

        # Save submission
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Head of submission:\n{df_sub.head()}")


def main():
    # Initialize Trainer
    trainer = Trainer()

    # Train
    trainer.fit()

    # Predict
    trainer.predict()


if __name__ == "__main__":
    main()
