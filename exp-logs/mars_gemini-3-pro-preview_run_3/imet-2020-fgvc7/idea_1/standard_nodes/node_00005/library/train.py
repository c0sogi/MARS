import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import seed_everything, calculate_micro_f1
from library.dataset import get_dataloaders
from library.model import ResNet50Classifier


class Trainer:
    """
    Trainer class to handle training, validation, and inference for the Artwork Classifier.
    """

    def __init__(self, device=None):
        self.config = Config
        self.device = device if device else torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = ResNet50Classifier(
            num_classes=self.config.NUM_CLASSES, pretrained=self.config.PRETRAINED
        )
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer & Scheduler (Initialized in fit)
        self.optimizer = None
        self.scheduler = None

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=self.config.USE_AMP)

        # State tracking
        self.best_val_f1 = 0.0
        self.best_threshold = self.config.DEFAULT_THRESHOLD

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=self.config.USE_AMP):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()

            # Unscale for gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            )

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader, threshold=None, use_tta=False):
        """
        Evaluates the model on the validation set.
        Returns average loss, F1 score, and raw predictions/targets.
        Cite solution_lesson_node_00004: Test Time Augmentation (TTA) for Inference Robustness.
        """
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_targets = []

        if threshold is None:
            threshold = self.config.DEFAULT_THRESHOLD

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(enabled=self.config.USE_AMP):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                    if use_tta:
                        # Horizontal Flip TTA
                        logits_flip = self.model(torch.flip(images, dims=[3]))
                        probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flip)) / 2
                    else:
                        probs = torch.sigmoid(logits)

                running_loss += loss.item()

                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / len(val_loader)

        # Concatenate all batches
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Calculate Metric
        f1 = calculate_micro_f1(all_probs, all_targets, threshold=threshold)

        return avg_loss, f1, all_probs, all_targets

    def optimize_threshold(self, probs, targets):
        """
        Finds the best threshold based on validation predictions to maximize F1.
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0.0
        best_thresh = self.config.DEFAULT_THRESHOLD

        for thresh in thresholds:
            score = calculate_micro_f1(probs, targets, threshold=thresh)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        return best_thresh, best_f1

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves to CSV.
        Cite solution_lesson_node_00004: Test Time Augmentation (TTA) for Inference Robustness.
        """
        print(f"Generating submission using threshold: {self.best_threshold:.4f}")
        self.model.eval()
        all_probs = []

        # We need IDs to map predictions back to files.
        # Since shuffle=False for test_loader, we can access them directly from the dataset.
        test_ids = test_loader.dataset.data["id"].values

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device, non_blocking=True)

                with autocast(enabled=self.config.USE_AMP):
                    logits = self.model(images)

                    # Horizontal Flip TTA
                    logits_flip = self.model(torch.flip(images, dims=[3]))
                    probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flip)) / 2

                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs).numpy()

        # Binarize predictions
        preds = (all_probs >= self.best_threshold).astype(int)

        # Format submission
        submission_rows = []
        for idx, row in enumerate(preds):
            image_id = test_ids[idx]
            # Get indices of active attributes
            attr_indices = np.where(row == 1)[0]
            attr_str = " ".join(map(str, attr_indices))
            submission_rows.append({"id": image_id, "attribute_ids": attr_str})

        # Save to CSV
        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

    def fit(self, debug=Config.DEBUG):
        """
        Main training loop.
        """
        seed_everything(self.config.SEED)

        # 1. Prepare Data
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            debug=debug,
        )

        # 2. Setup Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=self.config.EPOCHS,
        )

        print(f"Starting training on {self.device} for {self.config.EPOCHS} epochs.")

        # Early Stopping Parameters
        patience = 3
        patience_counter = 0

        # 3. Training Loop
        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_f1, _, _ = self.validate(
                val_loader, threshold=self.config.DEFAULT_THRESHOLD
            )

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val F1: {val_f1}, Time: {elapsed:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"New best model saved! F1 improved to {val_f1}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # 4. Post-Training: Threshold Optimization & Inference
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            print("Loading best model for inference...")
            self.model.load_state_dict(
                torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
            )

        # Optimize Threshold
        print("Optimizing threshold on validation set...")
        _, _, val_probs, val_targets = self.validate(val_loader)
        best_thresh, best_f1_opt = self.optimize_threshold(val_probs, val_targets)
        self.best_threshold = best_thresh
        print(f"Best Threshold: {self.best_threshold}, Optimized Val F1: {best_f1_opt}")
