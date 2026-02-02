import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import AverageMeter, save_checkpoint, load_checkpoint, set_seed
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import HCICNN


class Trainer:
    """
    Manages the training and validation process for the HCI-CNN model.
    """

    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for batch in loader:
            # Move data to device
            images = batch["image"].to(self.device)
            angles = batch["angle"].to(self.device)
            labels = batch["label"].to(self.device).view(-1, 1)

            # Forward pass
            self.optimizer.zero_grad()
            # HCICNN expects (x, angle)
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, loader):
        """
        Runs validation on the given loader.
        """
        self.model.eval()
        losses = AverageMeter()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                angles = batch["angle"].to(self.device)
                labels = batch["label"].to(self.device).view(-1, 1)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

        return losses.avg

    def fit(self, train_loader, val_loader, fold_idx):
        """
        Main training loop with Early Stopping.
        """
        best_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for Fold {fold_idx}...")

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print full precision metrics
            print(
                f"Fold {fold_idx} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Check for improvement
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best,
                fold_idx,
            )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered for Fold {fold_idx} at Epoch {epoch + 1}"
                )
                break

        print(f"Fold {fold_idx} finished. Best Val Loss: {best_loss}")
        return best_loss


def train_fold(fold_idx, load_cached_data=True):
    """
    Trains a single fold.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get data loaders
    train_loader, val_loader = get_fold_loaders(
        fold_idx, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = HCICNN().to(device)

    # Initialize Optimizer (AdamW with constant LR)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Create Trainer and Fit
    trainer = Trainer(model, device, optimizer, criterion)
    best_loss = trainer.fit(train_loader, val_loader, fold_idx)

    return best_loss


def train_all_folds(load_cached_data=True):
    """
    Sequentially trains all folds defined in Config.
    """
    fold_scores = []
    for fold_idx in range(Config.NUM_FOLDS):
        score = train_fold(fold_idx, load_cached_data=load_cached_data)
        fold_scores.append(score)

    print("\n--- Cross-Validation Results ---")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score}")
    print(f"Average Log Loss: {np.mean(fold_scores)}")


def generate_submission(load_cached_data=True):
    """
    Generates submission file by averaging predictions from all 5 fold models.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup_directories()

    print("Loading test data...")
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    fold_preds = []
    ids = []
    ids_collected = False

    # Iterate over all folds
    for fold_idx in range(Config.NUM_FOLDS):
        print(f"Predicting with model from Fold {fold_idx}...")

        # Instantiate and load model
        model = HCICNN().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )

        # Load weights
        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)
        model.eval()

        current_fold_probs = []
        current_ids = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)
                batch_ids = batch["id"]

                # Forward pass
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                current_fold_probs.extend(probs)

                if not ids_collected:
                    current_ids.extend(batch_ids)

        fold_preds.append(np.array(current_fold_probs))

        if not ids_collected:
            ids = current_ids
            ids_collected = True

    # Average predictions across folds
    avg_preds = np.mean(fold_preds, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": ids, "is_iceberg": avg_preds})

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
