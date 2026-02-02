import os
import torch
import torch.optim as optim
import numpy as np
import time
from library.config import Config, set_seed
from library.model import ICRCN
from library.loss import MultiStageLoss
from library.data_loader import get_loaders
from library.utils import (
    AverageMeter,
    compute_accuracy,
    compute_competition_metric,
    decode_predictions,
    apply_median_filter,
)


class Trainer:
    """
    Trainer class for the IC-RCN model.
    Manages training epochs, validation, and checkpointing.
    """

    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = ICRCN().to(self.device)

        # Initialize Loss
        self.criterion = MultiStageLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Training State
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        acc_gen = AverageMeter()
        acc_ref1 = AverageMeter()
        acc_ref2 = AverageMeter()

        for batch_idx, (features, targets, lengths) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward Pass
            # outputs is a dict: {'gen': ..., 'ref1': ..., 'ref2': ...}
            outputs = self.model(features)

            # Compute Loss
            total_loss, loss_dict = self.criterion(outputs, targets)

            # Backward Pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient Clipping (optional but good for LSTMs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update Metrics
            losses.update(total_loss.item(), features.size(0))

            # Compute accuracies for monitoring (ignoring padding -100)
            # Flatten for accuracy computation
            # outputs['gen']: (B, C, T) -> need to handle in compute_accuracy
            acc_gen.update(compute_accuracy(outputs["gen"], targets, ignore_index=-100))
            acc_ref1.update(
                compute_accuracy(outputs["ref1"], targets, ignore_index=-100)
            )
            acc_ref2.update(
                compute_accuracy(outputs["ref2"], targets, ignore_index=-100)
            )

        print(
            f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Train Loss: {losses.avg:.6f} | "
            f"Acc Gen: {acc_gen.avg:.4f} | Acc Ref1: {acc_ref1.avg:.4f} | Acc Ref2: {acc_ref2.avg:.4f}"
        )

        return losses.avg

    def validate(self):
        """
        Runs validation and computes the competition metric (Levenshtein Distance).
        """
        self.model.eval()
        losses = AverageMeter()

        all_preds_seq = []
        all_targets_seq = []

        with torch.no_grad():
            for features, targets, lengths in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward Pass
                outputs = self.model(features)

                # Compute Loss
                total_loss, _ = self.criterion(outputs, targets)
                losses.update(total_loss.item(), features.size(0))

                # Process Predictions for Metric Calculation
                # We use the final refinement stage (Ref2)
                logits = outputs["ref2"]  # (B, C, T)
                probs = torch.softmax(logits, dim=1)

                # Iterate over batch to handle variable lengths
                probs_np = probs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(features)):
                    length = lengths_np[i]

                    # Get relevant frames (remove padding)
                    # Shape: (C, T) -> (T, C) for processing
                    sample_probs = probs_np[i, :, :length].transpose(1, 0)
                    sample_target = targets_np[i, :length]

                    # 1. Apply Median Filter Smoothing
                    # apply_median_filter expects (T, C) or (T,)
                    smoothed_preds = apply_median_filter(sample_probs, kernel_size=5)

                    # 2. Decode to Sequence
                    pred_seq = decode_predictions(smoothed_preds)

                    # 3. Decode Target (Ground Truth)
                    # Targets are dense frame labels, need to collapse
                    target_seq = decode_predictions(sample_target)

                    all_preds_seq.append(pred_seq)
                    all_targets_seq.append(target_seq)

        # Compute Metric
        lev_score = compute_competition_metric(all_preds_seq, all_targets_seq)

        print(f"Validation Loss: {losses.avg:.6f} | Levenshtein Score: {lev_score}")

        return losses.avg, lev_score

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate()

            # Scheduler Step
            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            # Metric is Levenshtein distance (lower is better)
            if val_metric < self.best_metric:
                print(
                    f"Metric improved from {self.best_metric} to {val_metric}. Saving model..."
                )
                self.best_metric = val_metric
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Metric: {self.best_metric}")


def run_training():
    """
    Entry point to setup data and run the trainer.
    """
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Ensure directories exist
    Config.ensure_dirs()

    # Get Data Loaders
    # load_cached_data=True allows using pre-processed .npz files if they exist
    print("Loading data...")
    train_loader, val_loader, _, _ = get_loaders(load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Start Training
    trainer.train()
