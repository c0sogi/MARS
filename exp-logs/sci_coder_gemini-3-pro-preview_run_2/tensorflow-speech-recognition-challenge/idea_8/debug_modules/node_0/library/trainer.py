import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    NUM_EPOCHS,
    PATIENCE,
    WORKING_DIR,
)
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    save_checkpoint,
)
from library.model import MultiScaleEfficientNet


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device=DEVICE,
        working_dir=WORKING_DIR,
    ):
        """
        Args:
            model (nn.Module): The neural network to train.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            criterion (nn.Module): Loss function.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (torch.device): Device to run training on.
            working_dir (str): Directory to save checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.working_dir = working_dir
        self.best_acc = 0.0

        # Move model to device
        self.model.to(self.device)

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()

        losses = AverageMeter("Loss")
        accuracies = AverageMeter("Accuracy")

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

        return losses.avg, accuracies.avg

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()

        losses = AverageMeter("Val Loss")
        accuracies = AverageMeter("Val Accuracy")

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                acc = calculate_accuracy(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

        return losses.avg, accuracies.avg

    def fit(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
        """
        Main training loop with Early Stopping and Checkpointing.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        print(f"Starting training on {self.device} for {num_epochs} epochs...")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step Scheduler
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # Print metrics in full precision
            print(
                f"Epoch: {epoch}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Checkpointing and Early Stopping
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                patience_counter = 0
                print(f"New best validation accuracy: {self.best_acc}")
            else:
                patience_counter += 1

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "best_acc": self.best_acc,
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint,
                is_best,
                filename="last_checkpoint.pth",
                best_filename="best_model.pth",
            )

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")


def run_training(
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    learning_rate=LEARNING_RATE,
):
    """
    Helper function to setup the model, optimizer, and run training.
    """
    # 1. Initialize Model
    model = MultiScaleEfficientNet()

    # 2. Define Loss (CrossEntropy with Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # 3. Define Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    # 4. Define Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        working_dir=WORKING_DIR,
    )

    # 6. Run Training
    trainer.fit(train_loader, val_loader, num_epochs=num_epochs, patience=patience)

    return trainer.model
