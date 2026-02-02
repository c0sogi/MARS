import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    calculate_accuracy,
    AverageMeter,
    save_checkpoint,
    count_parameters,
)
from library.dataset import get_dataloaders, IDX2LABEL
from library.model import get_model


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction lifecycle
    of the Speech Command Recognition model.
    """

    def __init__(self):
        # 1. Setup Device and Seed
        set_seed(Config.seed)
        self.device = get_device()

        # 2. Initialize Model
        self.model = get_model()
        self.model.to(self.device)

        print(f"Model initialized: {Config.backbone}")
        print(f"Trainable parameters: {count_parameters(self.model)}")

        # 3. Loss Function
        # Label smoothing helps with generalization and calibration
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

        # 4. Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # 5. Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_max, eta_min=Config.eta_min
        )

        # 6. Training State
        self.best_acc = 0.0
        self.current_epoch = 0

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        start_time = time.time()

        for i, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc = calculate_accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

        # Update scheduler at the end of epoch
        self.scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch}/{Config.epochs}] Train | "
            f"Loss: {losses.avg:.6f} | Acc: {accuracies.avg:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        return losses.avg, accuracies.avg

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                acc = calculate_accuracy(outputs, labels)
                losses.update(loss.item(), images.size(0))
                accuracies.update(acc, images.size(0))

        print(
            f"Epoch [{self.current_epoch}/{Config.epochs}] Val   | "
            f"Loss: {losses.avg} | Acc: {accuracies.avg}"
        )

        return losses.avg, accuracies.avg

    def fit(self, debug=False):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training (Debug mode: {debug})...")

        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

        patience_counter = 0

        for epoch in range(1, Config.epochs + 1):
            self.current_epoch = epoch

            # Train
            self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                patience_counter = 0
                print(f"New best model found! Saving to {Config.best_model_path}")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "best_acc": self.best_acc,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=True,
                    filepath=Config.best_model_path,
                )
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.early_stopping_patience}"
                )

            # Early Stopping
            if patience_counter >= Config.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")

        # Run prediction on test set after training
        self.predict(test_loader)

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best model.
        """
        print("Starting inference on test set...")

        # Load best model
        if os.path.exists(Config.best_model_path):
            checkpoint = torch.load(Config.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            print(
                f"Loaded best model from {Config.best_model_path} (Epoch {checkpoint['epoch']}, Acc {checkpoint['best_acc']})"
            )
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model state."
            )

        self.model.eval()

        all_fnames = []
        all_preds = []

        with torch.no_grad():
            for images, _, fnames in test_loader:
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_fnames.extend(fnames)

        # Map indices to labels
        predicted_labels = [IDX2LABEL[idx] for idx in all_preds]

        # Create DataFrame
        submission_df = pd.DataFrame({"fname": all_fnames, "label": predicted_labels})

        # Save submission
        save_path = Config.submission_path
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(submission_df.head())
