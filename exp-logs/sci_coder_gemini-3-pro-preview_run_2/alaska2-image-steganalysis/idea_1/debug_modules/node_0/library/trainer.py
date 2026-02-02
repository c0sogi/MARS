import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, weighted_auc_score
from library.dataset import StegoDataset, get_transforms
from library.model import SRMEfficientNet


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with model, optimizer, criterion, and device settings.
        """
        # Reproducibility
        seed_everything(Config.SEED)

        self.device = torch.device(Config.DEVICE)

        # Model
        self.model = SRMEfficientNet(
            model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=Config.USE_AMP)

        # Best Score Tracker
        self.best_score = -float("inf")
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def get_dataloader(self, mode):
        """
        Creates and returns a DataLoader for the specified mode.
        """
        if mode == "train":
            csv_path = Config.TRAIN_CSV
            shuffle = True
            transform = get_transforms("train")
        elif mode == "val":
            csv_path = Config.VAL_CSV
            shuffle = False
            transform = get_transforms("val")
        elif mode == "test":
            csv_path = Config.TEST_CSV
            shuffle = False
            transform = get_transforms("test")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dataset = StegoDataset(
            csv_path=csv_path, mode=mode, transform=transform, load_cached_data=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=(mode == "train"),
        )

        return dataloader

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

            self.optimizer.zero_grad()

            with autocast(enabled=Config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation and calculates Weighted AUC.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

                with autocast(enabled=Config.USE_AMP):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Store predictions (sigmoid) and targets for metric calculation
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets = labels.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

        val_loss = running_loss / count

        # Concatenate all batches
        y_pred = np.concatenate(all_preds).ravel()
        y_true = np.concatenate(all_targets).ravel()

        # Calculate Weighted AUC
        score = weighted_auc_score(
            y_true,
            y_pred,
            tpr_thresholds=Config.TPR_THRESHOLDS,
            weights=Config.TPR_WEIGHTS,
        )

        return val_loss, score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Time: {elapsed:.2f}s - LR: {current_lr}"
            )
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss: {val_loss}")
            print(f"  Val Weighted AUC: {val_score}")

            # Early Stopping & Checkpointing
            if val_score > self.best_score:
                print(
                    f"  Score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"  Score did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Weighted AUC: {self.best_score}")

    def predict(self):
        """
        Generates predictions for the test set using TTA and saves to submission.csv.
        """
        print("Starting inference...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()
        test_loader = self.get_dataloader("test")

        ids = []
        preds = []

        # TTA: Test Time Augmentation (Original + Horizontal Flip)
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                # Get IDs from dataset based on current batch indices
                # The DataLoader is not shuffled, so order is preserved.
                # However, cleaner to rely on the order of iteration matching the dataset samples.
                start_idx = i * Config.BATCH_SIZE
                end_idx = start_idx + images.size(0)
                batch_samples = test_loader.dataset.samples[start_idx:end_idx]
                batch_ids = [s["image_id"] for s in batch_samples]
                ids.extend(batch_ids)

                images = images.to(self.device)

                # 1. Forward Pass: Original
                with autocast(enabled=Config.USE_AMP):
                    out_orig = self.model(images)

                # 2. Forward Pass: Horizontal Flip
                images_flipped = torch.flip(
                    images, dims=[3]
                )  # [B, C, H, W], dim 3 is Width
                with autocast(enabled=Config.USE_AMP):
                    out_flip = self.model(images_flipped)

                # Average Probabilities
                prob_orig = torch.sigmoid(out_orig)
                prob_flip = torch.sigmoid(out_flip)
                avg_prob = (prob_orig + prob_flip) / 2.0

                preds.extend(avg_prob.cpu().numpy().ravel())

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"Id": ids, "Label": preds})

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(submission_df.head())
