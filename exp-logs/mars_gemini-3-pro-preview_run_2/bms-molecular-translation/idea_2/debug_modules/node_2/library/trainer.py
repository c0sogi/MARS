import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    compute_levenshtein_distance,
)
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset
from library.model import Image2Seq


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction lifecycle
    of the Global Context Image-to-Sequence Network.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Components to be initialized in setup
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Data Loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # State
        self.start_epoch = 0
        self.best_metric = float("inf")  # Levenshtein distance (lower is better)

    def setup(self):
        """
        Prepares the datasets, tokenizer, model, and optimizer.
        """
        print("Setting up Trainer...")

        # 1. Load Metadata
        train_df = pd.read_csv(self.config.train_metadata_path)
        val_df = pd.read_csv(self.config.val_metadata_path)

        if self.config.debug:
            print("[DEBUG] limiting dataset size.")
            train_df = train_df.iloc[:1000]
            val_df = val_df.iloc[:200]

        # 2. Tokenizer
        self.tokenizer = Tokenizer(self.config)
        # Fit on training labels (handles caching internally)
        self.tokenizer.fit_on_texts(train_df["InChI"].values, load_cached_data=True)

        # 3. Datasets & Loaders
        train_dataset = InChiDataset(
            train_df, self.tokenizer, self.config, mode="train"
        )
        val_dataset = InChiDataset(val_df, self.tokenizer, self.config, mode="val")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # 4. Model
        print(f"Initializing model with vocab size: {self.tokenizer.vocab_size}")
        self.model = Image2Seq(self.config, self.tokenizer.vocab_size)
        self.model.to(self.device)

        # 5. Optimization
        self.optimizer = optim.Adam(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.config.encoder_lr,
                },
                {
                    "params": self.model.decoder.parameters(),
                    "lr": self.config.decoder_lr,
                },
                {
                    "params": self.model.init_h.parameters(),
                    "lr": self.config.decoder_lr,
                },
                {
                    "params": self.model.init_c.parameters(),
                    "lr": self.config.decoder_lr,
                },
            ],
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr,
        )

        # 6. Loss Function
        # Ignore padding token in loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        print("Setup complete.")

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        start_time = time.time()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            # labels shape: (Batch, Max_Len) -> <SOS> ... <EOS> <PAD>
            # We feed everything to the model, model expects to predict next token
            outputs = self.model(images, labels)

            # Targets: Shifted by 1 (exclude <SOS>)
            targets = labels[:, 1:]

            # Outputs shape: (Batch, Max_Len-1, Vocab_Size)
            # Flatten for CrossEntropyLoss
            loss = self.criterion(
                outputs.contiguous().view(-1, self.tokenizer.vocab_size),
                targets.contiguous().view(-1),
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )

            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch}][{i}/{len(self.train_loader)}] "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})"
                )

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Train Loss: {losses.avg:.15f} | Time: {elapsed:.2f}s")
        return losses.avg

    def validate(self):
        """
        Evaluates the model on the validation set using Levenshtein distance.
        """
        self.model.eval()
        distances = []
        start_time = time.time()

        print("Starting validation...")

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)

                # Generate predictions (Greedy Decoding)
                # Returns list of strings
                preds = self.model.predict(images, self.tokenizer)

                # Decode targets to strings for comparison
                targets = []
                for label_seq in labels:
                    target_str = self.tokenizer.sequence_to_text(label_seq)
                    targets.append(target_str)

                # Compute metric
                batch_dist = compute_levenshtein_distance(preds, targets)
                distances.append(batch_dist)

                if i % 50 == 0:
                    print(
                        f"Val Batch [{i}/{len(self.val_loader)}] Dist: {batch_dist:.4f}"
                    )

        mean_dist = np.mean(distances)
        elapsed = time.time() - start_time
        print(
            f"Validation Levenshtein Distance: {mean_dist:.15f} | Time: {elapsed:.2f}s"
        )

        return mean_dist

    def fit(self):
        """
        Main training loop with early stopping.
        """
        if self.model is None:
            self.setup()

        print(f"Starting training for {self.config.epochs} epochs...")

        no_improve_epochs = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metric = self.validate()

            # Scheduler Step
            self.scheduler.step(val_metric)

            # Checkpoint & Early Stopping
            is_best = val_metric < self.best_metric
            if is_best:
                self.best_metric = val_metric
                no_improve_epochs = 0
                print(f"New best metric: {self.best_metric:.15f}. Saving checkpoint.")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_metric": self.best_metric,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    is_best=True,
                    filepath=self.config.best_model_path,
                )
            else:
                no_improve_epochs += 1
                print(
                    f"No improvement for {no_improve_epochs} epochs. Best: {self.best_metric:.15f}"
                )

            if no_improve_epochs >= self.config.patience:
                print("Early stopping triggered.")
                break

    def predict(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Starting inference on test set...")

        # Load best model
        load_checkpoint(self.config.best_model_path, self.model, device=self.device)
        self.model.eval()

        # Prepare Test Loader
        test_df = pd.read_csv(self.config.test_metadata_path)
        if self.config.debug:
            test_df = test_df.iloc[:100]

        test_dataset = InChiDataset(test_df, self.tokenizer, self.config, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        results = []

        with torch.no_grad():
            for i, (images, image_ids) in enumerate(test_loader):
                images = images.to(self.device)

                # Predict
                preds = self.model.predict(images, self.tokenizer)

                for img_id, pred in zip(image_ids, preds):
                    results.append({"image_id": img_id, "InChI": pred})

                if i % 50 == 0:
                    print(f"Test Batch [{i}/{len(test_loader)}] processed.")

        # Save submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")
        print(f"Total predictions: {len(submission_df)}")
