import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import seed_everything, get_logger, calculate_score
from library.dataset import get_dataloaders
from library.model import AnimalEfficientNet


class Trainer:
    """
    Trainer class for the Animal Classification task.
    Handles training, validation, and inference loops.
    """

    def __init__(self, debug=Config.DEBUG):
        """
        Initialize the Trainer.

        Args:
            debug (bool): Whether to run in debug mode (subset of data).
        """
        seed_everything(Config.SEED)
        self.logger = get_logger("Trainer")
        self.device = Config.DEVICE
        self.debug = debug

        # Create working directory if it doesn't exist
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Load Data
        self.logger.info("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug, load_cached_data=True
        )

        # Initialize Model
        self.logger.info(f"Initializing model: {Config.MODEL_NAME}")
        self.model = AnimalEfficientNet(
            model_name=Config.MODEL_NAME,
            num_classes=Config.NUM_CLASSES,
            pretrained=Config.PRETRAINED,
            drop_path_rate=Config.DROP_PATH_RATE,
            use_gem=Config.USE_GEM_POOLING,
        )
        self.model.to(self.device)

        # Loss Function
        # Label smoothing is disabled in Config (0.0), but supported by CrossEntropyLoss if needed later
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler (OneCycleLR)
        # Total steps = epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.MAX_LR,
            epochs=Config.NUM_EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Checkpoint path
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler Step
            self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        elapsed = time.time() - start_time

        self.logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {avg_loss:.6f} - "
            f"Time: {elapsed:.2f}s"
        )

        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            score (float): Macro F1 score.
            avg_loss (float): Validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Mixed Precision Inference (optional but faster)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate and calculate score
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        score = calculate_score(all_labels, all_preds)

        return score, avg_loss

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_score = -float("inf")
        patience = 5  # Early stopping patience
        patience_counter = 0

        self.logger.info("Starting training...")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            val_score, val_loss = self.validate()

            self.logger.info(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val Macro F1: {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                self.logger.info(
                    f"New best score! Saving model to {self.checkpoint_path}"
                )
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training finished. Best Macro F1: {best_score}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        Applies Test-Time Augmentation (TTA) if configured.
        Saves submission file.
        """
        self.logger.info("Starting inference...")

        # Load best model
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded best model from {self.checkpoint_path}")
        else:
            self.logger.warning("No checkpoint found! Using current model weights.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, ids in self.test_loader:
                images = images.to(self.device, non_blocking=True)

                # TTA: Horizontal Flip
                if Config.TTA_FLIP:
                    # Forward pass original
                    with autocast():
                        logits_orig = self.model(images)

                        # Forward pass flipped
                        images_flipped = torch.flip(images, dims=[3])
                        logits_flip = self.model(images_flipped)

                        # Average logits
                        logits = (logits_orig + logits_flip) / 2.0
                else:
                    with autocast():
                        logits = self.model(images)

                preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Store results
                for img_id, pred in zip(ids, preds):
                    results.append({"Id": img_id, "Category": pred})

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Submission shape: {submission_df.shape}")
        self.logger.info(f"Head:\n{submission_df.head()}")
