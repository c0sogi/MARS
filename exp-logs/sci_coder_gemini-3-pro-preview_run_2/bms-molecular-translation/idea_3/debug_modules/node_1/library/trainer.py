import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import (
    tqdm,
)  # Not printing progress bars as per instructions, but good for structure logic

from library.config import Config
from library.utils import AverageMeter, calc_levenshtein, save_checkpoint
from library.tokenizer import InChITokenizer


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tokenizer = InChITokenizer()
        self.patience = patience

        # Optimization
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

        # State
        self.best_score = float("inf")
        self.current_patience = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        # Iterate over data
        # Note: Not using tqdm to avoid cluttering output as requested
        for i, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)  # (B, L)

            # Forward pass
            # labels contains [SOS, ..., EOS, PAD]
            # Model expects full sequence for teacher forcing logic inside forward
            logits = self.model(images, labels)  # (B, L-1, V)

            # Targets for loss are labels shifted by 1 (ignoring SOS)
            # labels[:, 1:] corresponds to [c1, c2, ..., EOS, PAD]
            targets = labels[:, 1:]  # (B, L-1)

            # Flatten for CrossEntropy
            loss = self.criterion(
                logits.reshape(-1, Config.VOCAB_SIZE), targets.reshape(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

            if Config.DEBUG and i >= 10:
                break

        return losses.avg

    def validate(self, epoch):
        """
        Runs validation loop to calculate Loss and Levenshtein distance.
        """
        self.model.eval()
        losses = AverageMeter()
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 1. Calculate Validation Loss (Teacher Forcing)
                logits = self.model(images, labels)
                targets = labels[:, 1:]
                loss = self.criterion(
                    logits.reshape(-1, Config.VOCAB_SIZE), targets.reshape(-1)
                )
                losses.update(loss.item(), images.size(0))

                # 2. Calculate Levenshtein Distance (Greedy Decoding)
                # Predict
                preds_indices = self.model.predict(
                    images, max_len=Config.MAX_LEN, device=self.device
                )

                # Decode
                for k in range(len(preds_indices)):
                    pred_str = self.tokenizer.decode(preds_indices[k])
                    # labels[k] includes SOS/EOS/PAD, tokenizer.decode handles stopping at EOS
                    true_str = self.tokenizer.decode(labels[k])

                    predictions.append(pred_str)
                    ground_truths.append(true_str)

                if Config.DEBUG and i >= 10:
                    break

        # Calculate metric
        score = calc_levenshtein(predictions, ground_truths)

        return losses.avg, score

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step(val_score)

            elapsed = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val Levenshtein: {val_score}"
            )

            # Checkpointing & Early Stopping
            is_best = val_score < self.best_score
            if is_best:
                self.best_score = val_score
                self.current_patience = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "best_score": self.best_score,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=True,
                )
                print(f"New best model saved with score: {self.best_score}")
            else:
                self.current_patience += 1
                print(f"Patience: {self.current_patience}/{self.patience}")

            if self.current_patience >= self.patience:
                print("Early stopping triggered.")
                break

    def generate_submission(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")
        self.model.eval()

        image_ids = []
        inchi_preds = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                images = batch["images"].to(self.device)
                ids = batch["image_ids"]

                # Predict
                preds_indices = self.model.predict(
                    images, max_len=Config.MAX_LEN, device=self.device
                )

                # Decode
                for k in range(len(preds_indices)):
                    pred_str = self.tokenizer.decode(preds_indices[k])
                    inchi_preds.append(pred_str)
                    image_ids.append(ids[k])

                if Config.DEBUG and i >= 5:
                    break

        # Create DataFrame
        df_sub = pd.DataFrame({"image_id": image_ids, "InChI": inchi_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save
        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"First 5 predictions:\n{df_sub.head()}")
