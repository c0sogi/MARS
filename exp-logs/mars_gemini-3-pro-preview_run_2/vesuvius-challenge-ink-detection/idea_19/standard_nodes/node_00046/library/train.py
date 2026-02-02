import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.model import SegFormerMiTB2
from library.loss import BCEDiceLoss
from library.dataset import InkDataset
from library.utils import set_seed


def calculate_fbeta(preds, targets, beta=0.5, threshold=0.5, smooth=1e-6):
    """
    Computes the F-Beta score.
    Args:
        preds (torch.Tensor): Logits or probabilities.
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): Beta value for F-score (0.5 weights precision higher).
        threshold (float): Threshold for binarization.
        smooth (float): Smoothing factor.
    Returns:
        float: F-Beta score.
    """
    # Apply sigmoid if logits (assuming input is logits based on model output)
    preds = torch.sigmoid(preds)

    # Binarize
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    # Flatten
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + (beta_sq * fn) + fp

    score = (numerator + smooth) / (denominator + smooth)
    return score.item()


class Trainer:
    def __init__(self, debug=False, epochs=Config.EPOCHS):
        self.debug = debug
        self.epochs = epochs
        self.device = torch.device(Config.DEVICE)
        self.best_score = -1.0
        self.save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        # Initialize Model
        self.model = SegFormerMiTB2()
        self.model.to(self.device)

        # Micro-Dataset Optimization Protocol: AdamW with conservative LR
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Loss Function
        self.criterion = BCEDiceLoss()

    def load_data(self):
        # Load Metadata
        df_train = pd.read_csv(Config.METADATA_TRAIN)
        df_val = pd.read_csv(Config.METADATA_VAL)

        if self.debug:
            print("Debug mode: Subsetting data...")
            df_train = df_train.head(16)
            df_val = df_val.head(16)

        # Initialize Datasets
        # load_cached_data=True ensures we use the caching mechanism in library/dataset.py
        train_dataset = InkDataset(df_train, split="train", load_cached_data=True)
        val_dataset = InkDataset(df_val, split="val", load_cached_data=True)

        # Initialize DataLoaders
        # Batch size strictly 8 as per protocol
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        print(
            f"Data loaded. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, labels, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            # masks (valid pixel masks) can be used for masking loss if needed,
            # but BCEDiceLoss in library/loss.py handles standard calc.
            # We assume labels are already 0 outside valid mask from preprocessing.

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        fbeta_scores = []

        with torch.no_grad():
            for images, labels, masks in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                # Calculate metric on valid pixels only?
                # The metric definition usually implies evaluating on the whole patch
                # or masked region. Here we calculate on the whole patch tensor
                # as the labels are 0 outside the mask.
                score = calculate_fbeta(
                    outputs, labels, beta=Config.BETA, threshold=Config.THRESHOLD
                )
                fbeta_scores.append(score)

        avg_score = np.mean(fbeta_scores)
        return avg_score

    def run(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_score = self.validate()

            print(
                f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.6f} | Val F0.5: {val_score}"
            )

            # Checkpointing
            if val_score > self.best_score:
                print(
                    f"Validation score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.save_path)
            else:
                print(f"Validation score did not improve (Best: {self.best_score}).")

        print(f"Training complete. Best Val F0.5: {self.best_score}")
        return self.best_score


def train_model(debug=False, epochs=Config.EPOCHS):
    """
    Main entry point to train the model.
    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    trainer = Trainer(debug=debug, epochs=epochs)
    trainer.load_data()
    best_score = trainer.run()

    return best_score
