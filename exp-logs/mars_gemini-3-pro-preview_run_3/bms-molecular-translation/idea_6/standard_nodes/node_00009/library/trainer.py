import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.utils import AverageMeter, compute_levenshtein


class Trainer:
    """
    Trainer class for the Formula-Conditioned Image-to-InChI model.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        tokenizer,
        criterion_text=None,
        criterion_formula=None,
        lambda_formula=0.5,
        checkpoint_dir="./working/idea_6/",
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            device (torch.device): Device to run training on.
            tokenizer (Tokenizer): Tokenizer for decoding/encoding.
            criterion_text (nn.Module): Loss function for text (default: CrossEntropy).
            criterion_formula (nn.Module): Loss function for formula (default: MSE).
            lambda_formula (float): Weight for the auxiliary formula loss.
            checkpoint_dir (str): Directory to save checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tokenizer = tokenizer

        # Initialize losses
        if criterion_text is None:
            # Ignore padding index in loss calculation
            self.criterion_text = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id
            )
        else:
            self.criterion_text = criterion_text

        if criterion_formula is None:
            self.criterion_formula = nn.MSELoss()
        else:
            self.criterion_formula = criterion_formula

        self.lambda_formula = lambda_formula
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        losses = AverageMeter()
        text_losses = AverageMeter()
        formula_losses = AverageMeter()

        for i, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            sequences = batch["sequence"].to(self.device)
            atom_counts = batch["atom_counts"].to(self.device)

            # Forward pass
            # Model expects sequences for teacher forcing
            logits, pred_atoms = self.model(images, sequences)

            # Text Loss Calculation
            # sequences: [SOS, c1, c2, ..., EOS, PAD]
            # input to model was: [SOS, c1, c2, ..., last_char] (handled inside model forward)
            # targets should be: [c1, c2, ..., EOS, PAD]
            targets = sequences[:, 1:]

            # Flatten for CrossEntropy: (Batch * SeqLen, VocabSize)
            loss_text = self.criterion_text(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

            # Formula Loss Calculation
            loss_formula = self.criterion_formula(pred_atoms, atom_counts)

            # Combined Loss
            loss = loss_text + self.lambda_formula * loss_formula

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            text_losses.update(loss_text.item(), batch_size)
            formula_losses.update(loss_formula.item(), batch_size)

        return losses.avg, text_losses.avg, formula_losses.avg

    def validate(self, val_loader):
        """
        Runs validation: computes loss and Levenshtein distance.
        """
        self.model.eval()

        losses = AverageMeter()
        levenshtein_distances = AverageMeter()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["image"].to(self.device)
                sequences = batch["sequence"].to(self.device)
                atom_counts = batch["atom_counts"].to(self.device)
                original_texts = batch["original_text"]

                # 1. Validation Loss (Teacher Forcing)
                logits, pred_atoms = self.model(images, sequences)
                targets = sequences[:, 1:]
                loss_text = self.criterion_text(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss_formula = self.criterion_formula(pred_atoms, atom_counts)
                loss = loss_text + self.lambda_formula * loss_formula
                losses.update(loss.item(), images.size(0))

                # 2. Metric Calculation (Greedy Decoding)
                # Predict sequence using the model's inference method
                predicted_strings = self.model.predict(
                    images, self.tokenizer, device=self.device
                )

                # Compute Levenshtein distance for the batch
                for pred_str, true_str in zip(predicted_strings, original_texts):
                    dist = compute_levenshtein(pred_str, true_str)
                    levenshtein_distances.update(dist)

        return losses.avg, levenshtein_distances.avg

    def fit(self, train_loader, val_loader, epochs, patience=5):
        """
        Main training loop with early stopping and checkpointing.
        """
        best_levenshtein = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on device {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train Phase
            train_loss, train_text_loss, train_formula_loss = self.train_epoch(
                train_loader, epoch
            )

            # Validation Phase
            val_loss, val_levenshtein = self.validate(val_loader)

            # Scheduler Step
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_levenshtein)
                else:
                    self.scheduler.step()

            elapsed = time.time() - start_time

            # Logging
            print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.1f}s")
            print(
                f"  Train Loss: {train_loss:.6f} (Text: {train_text_loss:.6f}, Formula: {train_formula_loss:.6f})"
            )
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val Levenshtein: {val_levenshtein:.6f}")

            # Checkpointing & Early Stopping
            if val_levenshtein < best_levenshtein:
                best_levenshtein = val_levenshtein
                patience_counter = 0
                self.save_checkpoint(epoch, val_levenshtein, is_best=True)
                print("  -> Best model saved!")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def save_checkpoint(self, epoch, metric, is_best=False):
        """
        Saves model checkpoint.
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metric": metric,
        }

        # Save latest checkpoint
        filename = "checkpoint.pth"
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, save_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(state, best_path)
