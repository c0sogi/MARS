import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from library.config import Config
from library.dataset import SpeechCommandsDataset
from library.model import EfficientNetV2Audio
from library.utils import (
    set_seed,
    AverageMeter,
    calculate_accuracy,
    ModelEMA,
    save_checkpoint,
    load_checkpoint,
)


class Trainer:
    """
    Trainer class for Speech Command Recognition.
    Handles training, validation, and inference using EfficientNetV2-B0 with EMA.
    """

    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # 1. Initialize Datasets
        self.train_dataset = SpeechCommandsDataset(subset="train")
        self.val_dataset = SpeechCommandsDataset(subset="val")

        # 2. Setup WeightedRandomSampler for Class Imbalance
        self.train_loader = self._create_train_loader()
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Initialize Model
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES)
        self.model.to(self.device)

        # 4. Initialize EMA
        # We use the EMA model for validation and inference
        self.ema = ModelEMA(self.model, decay=Config.EMA_DECAY, device=self.device)

        # 5. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def _create_train_loader(self):
        """
        Creates a DataLoader with WeightedRandomSampler to handle class imbalance.
        """
        # Extract labels from the dataset dataframe
        labels = self.train_dataset.df["label"].values

        # Calculate class counts
        class_counts = pd.Series(labels).value_counts()

        # Calculate weight per class (inverse frequency)
        class_weights = 1.0 / class_counts

        # Assign a weight to each sample
        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.DoubleTensor(sample_weights)

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")

        start_time = time.time()

        for i, (images, target) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Forward pass
            output = self.model(images)
            loss = self.criterion(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update EMA
            self.ema.update(self.model)

            # Metrics
            acc1 = calculate_accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch}/{Config.EPOCHS}] Train Loss: {losses.avg:.4f} | "
            f"Train Acc: {top1.avg:.4f}% | Time: {epoch_time:.2f}s"
        )

        return losses.avg, top1.avg

    def validate(self):
        """
        Runs validation using the EMA model.
        """
        # Use EMA model for validation
        eval_model = self.ema.get_model()
        eval_model.eval()

        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")

        with torch.no_grad():
            for images, target in enumerate(self.val_loader):
                # Unpack correctly (enumerate returns index, data)
                idx, (images, target) = images, target

                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = eval_model(images)
                loss = self.criterion(output, target)

                acc1 = calculate_accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))

        print(f"Validation Loss: {losses.avg} | Validation Acc: {top1.avg}%")
        return losses.avg, top1.avg

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_acc = 0.0
        patience_counter = 0

        print("Starting training...")

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Scheduler Step
            self.scheduler.step()

            # Checkpoint & Early Stopping
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
                print(f"New best accuracy: {best_acc}%")

                # Save best model (EMA weights)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.ema.get_model().state_dict(),
                        "best_score": best_acc,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    is_best=True,
                    filepath=Config.CHECKPOINT_PATH,
                )
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {best_acc}%")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Generating submission...")

        # Load Best Model
        model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES)
        model.to(self.device)

        # Load weights
        _, best_score = load_checkpoint(
            Config.CHECKPOINT_PATH, model, device=self.device
        )
        print(f"Loaded model with validation score: {best_score}%")

        model.eval()

        # Prepare Test Loader
        test_dataset = SpeechCommandsDataset(subset="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predictions = []
        fnames = test_dataset.df["fname"].tolist()

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                output = model(images)

                # Get predicted class indices
                _, preds = torch.max(output, 1)
                predictions.extend(preds.cpu().numpy())

        # Map indices to labels
        pred_labels = [Config.IDX2LABEL[idx] for idx in predictions]

        # create dataframe
        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model():
    """
    Entry point function to run the training and submission pipeline.
    """
    trainer = Trainer()
    trainer.fit()
    trainer.generate_submission()
