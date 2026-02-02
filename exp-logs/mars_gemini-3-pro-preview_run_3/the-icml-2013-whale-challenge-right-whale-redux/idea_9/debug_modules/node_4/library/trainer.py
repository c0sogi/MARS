import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import TrainConfig, ModelConfig
from library.utils import (
    get_device,
    AverageMeter,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)
from library.dataset import get_dataloaders
from library.model import get_model


class Trainer:
    """
    Manages the training, validation, and inference processes.
    """

    def __init__(self):
        self.device = get_device()
        set_seed(TrainConfig.seed)

        # Initialize Model
        self.model = get_model().to(self.device)

        # Initialize Loss Function
        # Using Inverse Class Frequency Weighting as per config
        pos_weight = torch.tensor([TrainConfig.bce_pos_weight]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TrainConfig.lr,
            weight_decay=TrainConfig.weight_decay,
        )

        # Scheduler will be initialized in fit() once epochs are confirmed
        self.scheduler = None

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch using Mixup with Mixed Losses.
        """
        self.model.train()
        losses = AverageMeter()

        for i, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            # Ensure labels are (B, 1) for BCEWithLogitsLoss
            labels = labels.to(self.device).view(-1, 1)

            batch_size = images.size(0)

            # Apply Mixup
            if TrainConfig.use_mixup and np.random.rand() < TrainConfig.mixup_prob:
                # Sample lambda from Beta distribution
                lam = np.random.beta(TrainConfig.mixup_alpha, TrainConfig.mixup_alpha)

                # Shuffle indices
                index = torch.randperm(batch_size).to(self.device)

                # Mix inputs
                mixed_images = lam * images + (1 - lam) * images[index, :]

                # Respective labels
                y_a, y_b = labels, labels[index]

                # Forward pass
                outputs = self.model(mixed_images)

                # Calculate Mixed Loss
                loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(
                    outputs, y_b
                )
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), batch_size)

        return losses.avg

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set and computes AUC.
        """
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()

                preds.extend(probs)
                targets.extend(labels.cpu().numpy())

        # Flatten arrays
        preds = np.array(preds).flatten()
        targets = np.array(targets).flatten()

        score = calculate_metrics(targets, preds)
        return score

    def fit(self, debug=False):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=True, debug=debug
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TrainConfig.epochs, eta_min=TrainConfig.min_lr
        )

        best_score = 0.0
        best_model_path = os.path.join(TrainConfig.working_dir, "best_model.pth")

        for epoch in range(TrainConfig.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Log metrics (full precision for AUC)
            print(
                f"Epoch {epoch+1}/{TrainConfig.epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val AUC: {val_score} - "
                f"Time: {elapsed:.2f}s"
            )

            # Save Checkpoint if improved
            if val_score > best_score:
                print(
                    f"Validation score improved ({best_score} -> {val_score}). Saving model..."
                )
                best_score = val_score
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    best_score,
                    best_model_path,
                )

        print(f"Training complete. Best Val AUC: {best_score}")

        # Generate Submission using the best model
        self.predict_submission(test_loader, best_model_path)

    def predict_submission(self, test_loader, model_path):
        """
        Loads the best model, predicts on the test set, and saves the submission file.
        """
        print(f"Loading best model from {model_path} for inference...")

        # Load weights
        load_checkpoint(self.model, model_path, device=self.device)
        self.model.eval()

        clips = []
        probs = []

        with torch.no_grad():
            for images, _, clip_names in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                clips.extend(clip_names)
                probs.extend(batch_probs)

        # Create Submission DataFrame
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df = pd.DataFrame({"clip": clips, "probability": probs})

        # Save to CSV
        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
