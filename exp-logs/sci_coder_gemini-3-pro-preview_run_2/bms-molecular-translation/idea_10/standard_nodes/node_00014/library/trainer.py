import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import seed_everything, AverageMeter, compute_levenshtein
from library.tokenizer import InChiTokenizer
from library.dataset import ChemicalImageDataset, ChemicalCollate
from library.model import HybridResNetTransformer


class Trainer:
    """
    Trainer class for the Anisotropic Hybrid ResNet-Transformer model.
    Handles training, validation, and checkpointing.
    """

    def __init__(self, config: Config):
        self.config = config
        seed_everything(config.seed)

        # 1. Initialize Tokenizer
        # Loads cached vocab if available, otherwise builds it from metadata
        self.tokenizer = InChiTokenizer(config, load_cached_data=True)

        # 2. Initialize Datasets and Dataloaders
        self.train_dataset = ChemicalImageDataset(config, self.tokenizer, mode="train")
        self.val_dataset = ChemicalImageDataset(config, self.tokenizer, mode="val")

        # Collate function handles variable-width padding
        collate_fn = ChemicalCollate(config, pad_id=self.tokenizer.PAD_ID)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        # 3. Initialize Model
        self.model = HybridResNetTransformer(config, self.tokenizer)
        self.model.to(config.device)

        # 4. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 5. Scheduler (OneCycleLR)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            steps_per_epoch=len(self.train_loader),
            epochs=config.epochs,
            pct_start=config.pct_start,
        )

        # State tracking
        self.best_score = float("inf")
        self.current_epoch = 0

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        ctc_losses = AverageMeter()
        ce_losses = AverageMeter()

        for step, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            lengths = batch["lengths"].to(self.config.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, targets=labels)

            # Calculate Joint Loss
            loss_dict = self.model.calc_loss(
                outputs, labels, lengths, ctc_weight=self.config.ctc_weight
            )

            loss = loss_dict["total"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            ctc_losses.update(loss_dict["ctc"].item(), batch_size)
            ce_losses.update(loss_dict["ce"].item(), batch_size)

            if (step + 1) % self.config.print_freq == 0:
                print(
                    f"Epoch: [{self.current_epoch + 1}][{step + 1}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} (CTC: {ctc_losses.avg:.4f}, CE: {ce_losses.avg:.4f}) "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )

        return losses.avg

    def validate(self):
        """
        Runs validation loop.
        Computes loss and Levenshtein distance.
        """
        self.model.eval()
        losses = AverageMeter()
        levenshtein_scores = []

        print("Starting validation...")

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                images = batch["images"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                lengths = batch["lengths"].to(self.config.device)

                # 1. Calculate Validation Loss (using Teacher Forcing)
                outputs = self.model(images, targets=labels)
                loss_dict = self.model.calc_loss(
                    outputs, labels, lengths, ctc_weight=self.config.ctc_weight
                )
                losses.update(loss_dict["total"].item(), images.size(0))

                # 2. Calculate Levenshtein Distance (Greedy Decoding)
                # model.predict returns a list of predicted InChI strings
                predictions = self.model.predict(images)

                # Decode ground truth labels for comparison
                targets = []
                for i in range(labels.size(0)):
                    # Extract sequence, ignoring padding/special tokens handled by decode()
                    target_indices = labels[i]
                    target_str = self.tokenizer.decode(target_indices)
                    targets.append(target_str)

                # Compute batch metric
                batch_score = compute_levenshtein(predictions, targets)
                levenshtein_scores.append(batch_score)

        avg_loss = losses.avg
        # Average Levenshtein score across all batches
        avg_levenshtein = (
            sum(levenshtein_scores) / len(levenshtein_scores)
            if levenshtein_scores
            else 0.0
        )

        return avg_loss, avg_levenshtein

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(
            f"Starting training for {self.config.epochs} epochs on device {self.config.device}"
        )

        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_one_epoch()

            # Validate
            val_loss, val_score = self.validate()

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss:.6f}")
            print(f"Val Levenshtein: {val_score}")  # Full precision

            # Checkpoint & Early Stopping
            if val_score < self.best_score:
                print(
                    f"Validation score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.config.model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation score did not improve. Patience: {patience_counter}/{self.config.patience}"
                )

            # Save checkpoint for resumption
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_score": self.best_score,
                },
                self.config.checkpoint_path,
            )

            if patience_counter >= self.config.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Levenshtein Score: {self.best_score}")


def run_training():
    """
    Entry point to run the training process.
    """
    config = Config()
    trainer = Trainer(config)
    trainer.fit()
