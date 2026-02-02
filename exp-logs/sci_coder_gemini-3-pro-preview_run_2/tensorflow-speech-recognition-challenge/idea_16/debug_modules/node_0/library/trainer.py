import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score

from library.config import Config, set_seed
from library.utils import (
    logger,
    save_checkpoint,
    load_checkpoint,
    ModelEMA,
    log_metrics,
)
from library.gpu_data_manager import GPUDataset
from library.model import EfficientNetV2Speech


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the Speech Command Recognition model.
    """

    def __init__(self):
        set_seed(Config.SEED)

        # 1. Data Setup
        logger.info("Initializing GPU Datasets...")
        self.train_data = GPUDataset(
            mode="train", device=Config.DEVICE, load_cached_data=True
        )
        self.val_data = GPUDataset(
            mode="val", device=Config.DEVICE, load_cached_data=True
        )

        # 2. Model Setup
        logger.info(f"Initializing Model: {Config.MODEL_NAME}")
        self.model = EfficientNetV2Speech(pretrained=True)
        self.model.to(Config.DEVICE)

        # 3. EMA Setup
        # We use EMA for validation and inference to improve stability
        self.ema = ModelEMA(self.model, decay=Config.EMA_DECAY, device=Config.DEVICE)

        # 4. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # 5. State
        self.start_epoch = 0
        self.best_acc = 0.0

        # Attempt to resume
        epoch, metrics = load_checkpoint(
            self.model, self.optimizer, filename="last_checkpoint.pth"
        )
        if epoch is not None:
            self.start_epoch = epoch + 1
            if metrics and "val_acc" in metrics:
                self.best_acc = metrics["val_acc"]
            # Sync EMA with loaded model if resuming
            self.ema = ModelEMA(
                self.model, decay=Config.EMA_DECAY, device=Config.DEVICE
            )

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()

        # Calculate iterations based on dataset size
        num_samples = len(self.train_data)
        num_batches = num_samples // Config.BATCH_SIZE

        running_loss = 0.0
        all_preds = []
        all_targets = []

        for _ in range(num_batches):
            # Get batch from GPU Dataset (Weighted Random Sampling)
            waveforms, labels = self.train_data.get_batch(Config.BATCH_SIZE)

            self.optimizer.zero_grad()

            # Forward pass with Noise Injection (passing noise_bank)
            # The frontend handles augmentation internally
            logits = self.model(waveforms, noise_bank=self.train_data.noise_bank)

            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            # Update EMA
            if Config.USE_EMA:
                self.ema.update(self.model)

            # Metrics tracking
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

        # Step scheduler
        self.scheduler.step()

        avg_loss = running_loss / num_batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = accuracy_score(all_targets, all_preds)

        return {"train_loss": avg_loss, "train_acc": acc}

    @torch.no_grad()
    def validate(self, dataset):
        """
        Runs validation using the EMA model.
        """
        # Use EMA model for validation
        eval_model = self.ema.ema_model
        eval_model.eval()

        running_loss = 0.0
        all_preds = []
        all_targets = []

        # Iterate sequentially
        batch_iterator = dataset.get_iterator(Config.BATCH_SIZE)
        num_batches = 0

        for waveforms, labels in batch_iterator:
            # Forward pass without noise bank (no noise injection)
            logits = eval_model(waveforms, noise_bank=None)

            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            num_batches += 1

        if num_batches == 0:
            return {"val_loss": 0.0, "val_acc": 0.0}

        avg_loss = running_loss / num_batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = accuracy_score(all_targets, all_preds)

        return {"val_loss": avg_loss, "val_acc": acc}

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        logger.info(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        patience = 5
        patience_counter = 0

        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate(self.val_data)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch

            # Log
            logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            log_metrics(metrics)

            # Checkpoint Logic
            current_acc = val_metrics["val_acc"]

            # Save last checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                metrics,
                filename="last_checkpoint.pth",
            )

            # Save best checkpoint (using EMA weights)
            if current_acc > self.best_acc:
                self.best_acc = current_acc
                patience_counter = 0
                logger.info(f"New best accuracy: {self.best_acc}. Saving best model.")
                # Save the EMA model as the best model
                save_checkpoint(
                    self.ema,  # Pass EMA wrapper to save its internal model
                    self.optimizer,
                    epoch,
                    metrics,
                    filename="best_model.pth",
                )
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        logger.info(f"Training finished. Best Validation Accuracy: {self.best_acc}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        logger.info("Loading best model for submission...")

        # Load test data
        test_data = GPUDataset(mode="test", device=Config.DEVICE, load_cached_data=True)

        # Load best weights into the main model (or a fresh instance)
        # We use a fresh instance to ensure clean state, wrapped in EMA structure if needed
        # But load_checkpoint handles loading into the provided model.
        # Since we saved the EMA model as "best_model.pth", we load it into self.model
        # to use for inference.
        epoch, metrics = load_checkpoint(self.model, filename="best_model.pth")

        self.model.eval()

        all_preds = []
        all_fnames = []

        # We need fnames. GPUDataset doesn't store fnames in the tensor cache.
        # We must read the test csv to get the order of files, which matches the sequential iterator.
        df_test = pd.read_csv(Config.TEST_CSV)
        fnames = df_test["fname"].tolist()

        logger.info("Generating predictions...")

        batch_iterator = test_data.get_iterator(Config.BATCH_SIZE)

        with torch.no_grad():
            for waveforms, _ in batch_iterator:
                # Forward pass
                logits = self.model(waveforms, noise_bank=None)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Map IDs to Labels
        pred_labels = [Config.ID2LABEL[idx] for idx in all_preds]

        # Create Submission DataFrame
        # Ensure lengths match
        if len(pred_labels) != len(fnames):
            logger.error(
                f"Mismatch: {len(pred_labels)} predictions vs {len(fnames)} files."
            )
            # Truncate or pad if necessary (though shouldn't happen with correct logic)
            min_len = min(len(pred_labels), len(fnames))
            pred_labels = pred_labels[:min_len]
            fnames = fnames[:min_len]

        df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
