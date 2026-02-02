import os
import time
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein, save_checkpoint


class Trainer:
    """
    Trainer class for the InChI recognition model.
    Handles training, validation, and checkpointing.
    """

    def __init__(
        self, model, tokenizer, optimizer, scheduler=None, device=Config.DEVICE
    ):
        """
        Args:
            model (nn.Module): The Seq2Seq model.
            tokenizer (Tokenizer): Tokenizer for decoding.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            device (torch.device): Device to run on.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        # Ignore padding index in loss calculation
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.char_to_idx[Config.PAD_TOKEN]
        )

    def train_epoch(self, train_loader, epoch, debug=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        losses = AverageMeter()

        # Determine number of steps for debugging
        steps = len(train_loader)
        if debug:
            steps = min(steps, 10)

        for i, (images, labels, _) in enumerate(train_loader):
            if i >= steps:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Forward pass
            # labels are passed for teacher forcing
            outputs = self.model(images, labels)

            # Calculate loss
            # outputs: [batch, seq_len, vocab_size]
            # labels: [batch, seq_len]
            # We discard the first time step (SOS) from outputs and labels for loss
            # outputs[:, 0, :] is all zeros (unused)
            # We want outputs[:, t, :] to predict labels[:, t]
            output_logits = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            target_labels = labels[:, 1:].reshape(-1)

            loss = self.criterion(output_logits, target_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            self.optimizer.step()

            losses.update(loss.item(), batch_size)

            if i % 100 == 0:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}] Loss: {losses.avg:.4f}"
                )

        return losses.avg

    def validate(self, val_loader, debug=False):
        """
        Evaluates the model on the validation set using Levenshtein distance.
        """
        self.model.eval()
        levenshtein_meter = AverageMeter()

        steps = len(val_loader)
        if debug:
            steps = min(steps, 10)

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(val_loader):
                if i >= steps:
                    break

                images = images.to(self.device)

                # Generate predictions using greedy decoding
                preds = self.model.predict(images, self.tokenizer)

                # Convert ground truth tensor to strings
                targets = []
                for label in labels:
                    targets.append(self.tokenizer.sequence_to_text(label))

                # Compute metric
                score = compute_levenshtein(preds, targets)
                levenshtein_meter.update(score, images.size(0))

        return levenshtein_meter.avg

    def fit(self, train_loader, val_loader, epochs, patience=3, debug=Config.DEBUG):
        """
        Main training loop with early stopping.
        """
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nStarting Epoch {epoch+1}/{epochs}")
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch, debug)

            # Validate
            val_score = self.validate(val_loader, debug)

            # Scheduler step
            if self.scheduler:
                # ReduceLROnPlateau expects a metric
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_score)
                else:
                    self.scheduler.step()

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1} - Train Loss: {train_loss:.6f} - Val Levenshtein: {val_score:.6f} - Time: {elapsed:.0f}s"
            )

            # Save checkpoint and Early Stopping
            is_best = val_score < best_metric
            if is_best:
                best_metric = val_score
                patience_counter = 0
                print(
                    f"New best model found (Levenshtein: {best_metric:.6f}). Saving checkpoint..."
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    "best_metric": best_metric,
                },
                is_best,
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break


def generate_submission(
    model, test_loader, tokenizer, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    device = Config.DEVICE
    model.to(device)

    image_ids = []
    inchi_preds = []

    print(f"\nGenerating submission for {len(test_loader.dataset)} samples...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        for i, (images, ids) in enumerate(test_loader):
            images = images.to(device)

            # Predict
            preds = model.predict(images, tokenizer)

            image_ids.extend(ids)
            inchi_preds.extend(preds)

            if i % 100 == 0:
                print(f"Processed {i}/{len(test_loader)} batches")

    # Create submission DataFrame
    df = pd.DataFrame({"image_id": image_ids, "InChI": inchi_preds})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
