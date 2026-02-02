import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    LevenshteinMetric,
    save_checkpoint,
    load_checkpoint,
)
from library.model import Seq2Seq
from library.dataset import get_dataloader
from library.tokenizer import Tokenizer


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction lifecycle
    of the Chemical Image to InChI translation model.
    """

    def __init__(self, load_cached_data=True):
        # 1. Setup Device
        self.device = torch.device(Config.DEVICE)
        print(f"Trainer initialized on device: {self.device}")

        # 2. Initialize Tokenizer
        self.tokenizer = Tokenizer(load_cached_data=load_cached_data)
        self.vocab_size = len(self.tokenizer)

        # 3. Initialize Model
        self.model = Seq2Seq(self.vocab_size).to(self.device)

        # 4. Optimizer & Loss
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Ignore padding index in loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD_IDX)

        # 5. Load Checkpoint if exists
        self.start_epoch, self.best_score = load_checkpoint(
            self.model, self.optimizer, filename="checkpoint.pth"
        )

    def fit(self, epochs=Config.EPOCHS, debug=Config.DEBUG):
        """
        Main training loop with validation and early stopping.
        """
        print(f"\nStarting training for {epochs} epochs (Debug={debug})...")

        # Data Loaders
        train_loader = get_dataloader(
            Config.TRAIN_METADATA, self.tokenizer, mode="train", debug=debug
        )
        val_loader = get_dataloader(
            Config.VAL_METADATA, self.tokenizer, mode="val", shuffle=False, debug=debug
        )

        patience = 5
        patience_counter = 0

        for epoch in range(self.start_epoch, epochs):
            # --- Training ---
            train_loss = self.train_epoch(train_loader, epoch)

            # --- Validation ---
            val_loss, val_score = self.validate(val_loader)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Levenshtein: {val_score:.4f}"
            )

            # --- Checkpointing & Early Stopping ---
            is_best = val_score < self.best_score
            if is_best:
                self.best_score = val_score
                patience_counter = 0
                print(f"New best score: {self.best_score:.4f}. Saving model...")
            else:
                patience_counter += 1
                print(f"Score did not improve. Patience: {patience_counter}/{patience}")

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_score": self.best_score,
                },
                is_best,
            )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def train_epoch(self, loader, epoch):
        """
        Executes one training epoch.
        """
        self.model.train()
        losses = AverageMeter("Loss", ":.4f")

        for i, (images, labels, label_lengths) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            # teacher_forcing_ratio controls how often we feed the true previous token
            outputs = self.model(
                images, labels, teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO
            )
            # outputs: (B, max_len, vocab_size)
            # labels: (B, max_len)

            # Reshape for CrossEntropyLoss
            # We skip the first time step (t=0) because it corresponds to <SOS> input
            # and the first output is the prediction for the second token.
            # labels[:, 1:] are the targets for outputs[:, 1:]
            output_dim = outputs.shape[-1]

            # Flatten outputs and targets
            # outputs[:, 1:] -> (B, max_len-1, vocab_size) -> (B * (max_len-1), vocab_size)
            # labels[:, 1:]  -> (B, max_len-1)             -> (B * (max_len-1))
            loss = self.criterion(
                outputs[:, 1:].reshape(-1, output_dim), labels[:, 1:].reshape(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

            if i % Config.PRINT_FREQ == 0:
                print(
                    f"Epoch: [{epoch + 1}][{i}/{len(loader)}] Loss {losses.val:.4f} ({losses.avg:.4f})"
                )

        return losses.avg

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Computes Loss and Levenshtein Distance.
        """
        self.model.eval()
        losses = AverageMeter("Val Loss", ":.4f")
        metric = LevenshteinMetric()

        with torch.no_grad():
            for i, (images, labels, label_lengths) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 1. Calculate Validation Loss
                # We use ratio=0.0 to mimic inference behavior for loss calculation context,
                # or we can use 1.0 to measure perplexity against ground truth.
                # Here we use 0.0 to see how well the model does on its own predictions vs targets.
                outputs = self.model(images, labels, teacher_forcing_ratio=0.0)

                output_dim = outputs.shape[-1]
                loss = self.criterion(
                    outputs[:, 1:].reshape(-1, output_dim), labels[:, 1:].reshape(-1)
                )
                losses.update(loss.item(), images.size(0))

                # 2. Calculate Levenshtein Metric
                # Use greedy decoding for metric calculation
                preds = self.model.predict(images)

                # Decode strings
                decoded_preds = [self.tokenizer.decode(p) for p in preds]
                decoded_targets = [self.tokenizer.decode(l) for l in labels]

                metric.update(decoded_preds, decoded_targets)

        return losses.avg, metric.get_avg_score()

    def predict_test(self, debug=Config.DEBUG):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("\n--- Generating Predictions for Test Set ---")

        # Load best model weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading best model from {Config.BEST_MODEL_PATH}")
            checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            print("Warning: No best model found. Using current model weights.")

        self.model.eval()

        test_loader = get_dataloader(
            Config.TEST_METADATA,
            self.tokenizer,
            mode="test",
            shuffle=False,
            debug=debug,
        )

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for i, (images, image_ids) in enumerate(test_loader):
                images = images.to(self.device)

                # Greedy decoding
                pred_indices = self.model.predict(images)

                # Decode to strings
                decoded_preds = [self.tokenizer.decode(p) for p in pred_indices]

                all_preds.extend(decoded_preds)
                all_ids.extend(image_ids)

                if i % Config.PRINT_FREQ == 0:
                    print(f"Processed {i}/{len(test_loader)} batches")

        # Create submission dataframe
        submission_df = pd.DataFrame({"image_id": all_ids, "InChI": all_preds})

        # Save
        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(f"Total predictions: {len(submission_df)}")
