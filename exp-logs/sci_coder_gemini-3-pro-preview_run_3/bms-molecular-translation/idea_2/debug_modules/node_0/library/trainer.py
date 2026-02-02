import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import (
    AverageMeter,
    compute_levenshtein,
    save_checkpoint,
    load_checkpoint,
)
from library.model import InChiModel
from library.dataset import get_dataloaders
from library.tokenizer import Tokenizer


class Trainer:
    """
    Trainer class to handle training, validation, and prediction for InChI recognition.
    """

    def __init__(self, load_cached_data=True, debug=False):
        """
        Initialize the Trainer.

        Args:
            load_cached_data (bool): Whether to load cached vocabulary.
            debug (bool): If True, use a small subset of data for debugging.
        """
        # 1. Setup Tokenizer
        self.tokenizer = Tokenizer(load_cached_data=load_cached_data)

        # 2. Setup DataLoaders
        subset_size = 1000 if debug else None
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            debug_subset_size=subset_size,
        )

        # 3. Setup Model
        self.model = InChiModel(vocab_size=len(self.tokenizer))
        self.model.to(Config.DEVICE)

        # 4. Setup Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Setup Scheduler
        # OneCycleLR requires the total number of steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(self.train_loader),
            epochs=Config.NUM_EPOCHS,
            pct_start=0.1,
        )

        # 6. Setup Loss Function
        # We ignore the padding token in the loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # 7. Mixed Precision Scaler
        self.scaler = GradScaler()

        # State
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        batch_time = AverageMeter()
        start = time.time()

        print(f"\n[Epoch {epoch + 1}/{Config.NUM_EPOCHS}] Training...")

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)

            # Teacher Forcing:
            # Input to decoder: sequence excluding the last token (EOS or Pad)
            # Target: sequence excluding the first token (SOS)
            # Shapes: (B, Seq_Len)
            decoder_input = labels[:, :-1]
            targets = labels[:, 1:]

            self.optimizer.zero_grad()

            with autocast():
                # Forward pass
                # logits: (B, Seq_Len - 1, Vocab_Size)
                logits = self.model(
                    images, decoder_input, pad_token_id=self.tokenizer.pad_token_id
                )

                # Reshape for CrossEntropyLoss: (B * (Seq_Len - 1), Vocab_Size) vs (B * (Seq_Len - 1))
                loss = self.criterion(
                    logits.reshape(-1, len(self.tokenizer)), targets.reshape(-1)
                )

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step Scheduler
            self.scheduler.step()

            # Logging
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % 100 == 0:
                print(
                    f"Batch {i}/{len(self.train_loader)} | "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )

        print(f"Epoch {epoch + 1} Train Loss: {losses.avg:.6f}")
        return losses.avg

    def validate(self, epoch):
        """
        Runs validation loop.
        """
        self.model.eval()
        losses = AverageMeter()

        print(f"\n[Epoch {epoch + 1}/{Config.NUM_EPOCHS}] Validating...")

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)

                decoder_input = labels[:, :-1]
                targets = labels[:, 1:]

                with autocast():
                    logits = self.model(
                        images, decoder_input, pad_token_id=self.tokenizer.pad_token_id
                    )
                    loss = self.criterion(
                        logits.reshape(-1, len(self.tokenizer)), targets.reshape(-1)
                    )

                losses.update(loss.item(), images.size(0))

        print(f"Epoch {epoch + 1} Val Loss: {losses.avg:.10f}")
        return losses.avg

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {Config.DEVICE}...")
        patience_counter = 0

        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            _ = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Checkpoint Logic
            is_best = val_loss < self.best_val_loss
            if is_best:
                print(
                    f"Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f}. Saving checkpoint."
                )
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "best_metric": self.best_val_loss,
                },
                is_best=is_best,
            )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    def predict_test(self):
        """
        Generates predictions for the test set and saves submission file.
        """
        print("\n--- Generating Predictions for Test Set ---")

        # Load best model
        best_model_path = os.path.join(
            Config.WORKING_DIR, "checkpoints", "model_best.pth.tar"
        )
        load_checkpoint(best_model_path, self.model)
        self.model.eval()

        predictions = []
        image_ids = []

        # We can use the test loader from get_dataloaders
        # Note: The test loader returns (image, dummy_label). We only need images.
        # We need to access the image_ids. The dataset returns (image, label_seq).
        # We need to look up image_ids from the dataframe in the dataset.

        test_df = self.test_loader.dataset.df

        print(f"Predicting {len(test_df)} samples...")

        # To map predictions back to IDs, we iterate sequentially.
        # DataLoader is set to shuffle=False for test_loader in dataset.py.

        with torch.no_grad():
            for i, (images, _) in enumerate(self.test_loader):
                images = images.to(Config.DEVICE)

                # Greedy decoding
                # Output shape: (B, Seq_Len)
                pred_seqs = self.model.predict(
                    images, self.tokenizer, max_len=Config.MAX_LEN
                )

                # Convert sequences to text
                for seq in pred_seqs:
                    inchi_str = self.tokenizer.sequence_to_text(seq)
                    predictions.append(inchi_str)

                if i % 50 == 0:
                    print(f"Processed batch {i}/{len(self.test_loader)}")

        # Verify alignment
        if len(predictions) != len(test_df):
            print(
                f"Warning: Prediction count {len(predictions)} != Metadata count {len(test_df)}"
            )
            # Truncate or pad if necessary, though this shouldn't happen with correct dataloader
            if len(predictions) > len(test_df):
                predictions = predictions[: len(test_df)]

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"image_id": test_df["image_id"].values, "InChI": predictions}
        )

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generation complete.")
