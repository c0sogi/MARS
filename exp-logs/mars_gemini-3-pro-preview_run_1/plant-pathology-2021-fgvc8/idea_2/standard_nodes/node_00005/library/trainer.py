import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, Logger, calculate_f1_score, save_checkpoint
from library.model import create_model
from library.dataset import get_loaders, get_test_loader


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.logger = Logger()

        # Data Loaders
        self.train_loader, self.val_loader = get_loaders()

        # Model
        self.model = create_model(pretrained=True)

        # Loss Function (BCEWithLogitsLoss works with soft targets from Mixup)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

        self.best_score = 0.0
        self.early_stopping_patience = 5
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Validation does not use Mixup, so targets are binary here
                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                losses.update(loss.item(), images.size(0))

                # Store predictions and targets for F1 calculation
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        score = calculate_f1_score(all_preds, all_targets, threshold=Config.THRESHOLD)

        return losses.avg, score

    def fit(self):
        self.logger.log(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate(epoch)

            # Update Scheduler
            self.scheduler.step()

            # Log metrics
            self.logger.log(
                f"Epoch {epoch}/{Config.EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val F1: {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_score,
                    "best_model.pth",
                )
                self.logger.log(f"New best model saved with F1: {val_score}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                self.logger.log(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.log(f"Training complete. Best F1 Score: {self.best_score}")

    def predict(self):
        self.logger.log("Starting inference on test set...")

        # Load best model
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            self.logger.log("No checkpoint found! Using current model weights.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.logger.log(f"Loaded checkpoint from {checkpoint_path}")

        self.model.eval()
        test_loader = get_test_loader()

        all_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    outputs = self.model(images)

                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > Config.THRESHOLD).int().cpu().numpy()
                all_preds.append(preds)

        final_preds = np.vstack(all_preds)

        # Convert binary predictions to labels
        # We need the test metadata to get the image IDs in the correct order
        # DataLoader with shuffle=False preserves order
        df_test = pd.read_csv(Config.TEST_METADATA)
        image_ids = df_test["image"].values

        submission_rows = []
        for idx, row_preds in enumerate(final_preds):
            image_id = image_ids[idx]

            # Get list of classes where prediction is 1
            labels = [Config.CLASSES[i] for i, p in enumerate(row_preds) if p == 1]

            # If no label is predicted, it's usually considered "healthy" or handled by threshold
            # However, the problem statement implies 'healthy' is a specific class in the list.
            # If the model predicts nothing (all zeros), we join an empty list -> empty string.
            # In this dataset, 'healthy' is explicit.
            label_str = " ".join(labels)

            # Fallback if empty? (Optional, depending on competition rules, usually empty string is valid or implies healthy if healthy class exists)
            # Given 'healthy' is a class, if it's not predicted, the leaf is likely unhealthy but confidence was low.
            # We stick to the model's output.

            submission_rows.append({"image": image_id, "labels": label_str})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.log(f"Submission saved to {Config.SUBMISSION_PATH}")
