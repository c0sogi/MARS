import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, AverageMeter, calculate_auc, get_logger
from library.models import get_model
from library.data import get_dataloaders


class Trainer:
    """
    Manages the training lifecycle for a single model.
    """

    def __init__(self, model_name, dataloaders, device=Config.DEVICE):
        self.model_name = model_name
        self.dataloaders = dataloaders
        self.device = device

        # Initialize Logger
        log_path = os.path.join(Config.WORKING_DIR, f"{model_name}_train.log")
        self.logger = get_logger(f"{model_name}_trainer", log_path)

        # Initialize Model
        self.logger.info(f"Initializing model: {model_name}")
        self.model = get_model(
            model_name, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES
        )
        self.model.to(self.device)

        # Loss Function (Binary Classification with Logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # State tracking
        self.best_auc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        losses = AverageMeter()

        # Containers for epoch-level metric calculation
        all_targets = []
        all_preds = []

        for i, (images, labels) in enumerate(self.dataloaders["train"]):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1, 1)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Record metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)

            # Store predictions for AUC
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_targets.extend(labels.detach().cpu().numpy())
            all_preds.extend(probs)

        # Calculate training AUC
        train_auc = calculate_auc(all_targets, all_preds)

        self.logger.info(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] Train Loss: {losses.avg:.6f} | Train AUC: {train_auc}"
        )
        return losses.avg, train_auc

    def validate(self, epoch):
        """Runs validation."""
        self.model.eval()
        losses = AverageMeter()

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, labels in enumerate(self.dataloaders["val"]):
                # Unpack correctly (enumerate returns index, data)
                # But dataloader yields (images, labels)
                # Correcting loop structure:
                pass

            for i, (images, labels) in enumerate(self.dataloaders["val"]):
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                losses.update(loss.item(), batch_size)

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(probs)

        val_auc = calculate_auc(all_targets, all_preds)

        self.logger.info(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] Val Loss: {losses.avg:.6f} | Val AUC: {val_auc}"
        )
        return losses.avg, val_auc

    def fit(self):
        """Main training loop with Early Stopping."""
        self.logger.info(f"Starting training for {self.model_name}...")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            self.train_epoch(epoch)

            # Validate
            _, val_auc = self.validate(epoch)

            # Update Scheduler
            self.scheduler.step()

            # Checkpoint & Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                patience_counter = 0

                # Save best model
                save_path = os.path.join(
                    Config.WORKING_DIR, f"{self.model_name}_best.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"New best model saved with AUC: {val_auc}")
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            elapsed = time.time() - start_time
            self.logger.info(f"Epoch time: {elapsed:.2f}s")

            if patience_counter >= Config.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(
            f"Training finished for {self.model_name}. Best AUC: {self.best_auc} at Epoch {self.best_epoch+1}"
        )
        return self.best_auc


def run_training(debug=False, sample_size=1000):
    """
    Orchestrates the training of the heterogeneous ensemble.

    Args:
        debug (bool): If True, runs on a subset of data.
        sample_size (int): Size of subset if debug is True.
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Get DataLoaders
    # Note: We load data once and reuse for both models to ensure identical splits/shuffles if any
    print("Loading data...")
    dataloaders = get_dataloaders(
        train_path=Config.TRAIN_METADATA_PATH,
        val_path=Config.VAL_METADATA_PATH,
        test_path=Config.TEST_METADATA_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        sample_size=sample_size,
    )

    results = {}

    # Train each model in the ensemble
    for model_name in Config.MODEL_NAMES:
        print(f"\n{'='*40}")
        print(f"Training Ensemble Member: {model_name}")
        print(f"{'='*40}\n")

        trainer = Trainer(model_name, dataloaders)
        best_auc = trainer.fit()
        results[model_name] = best_auc

        # Clear GPU memory between models
        del trainer
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print("Ensemble Training Complete")
    print("=" * 40)
    for name, auc in results.items():
        print(f"{name}: Best Val AUC = {auc}")
