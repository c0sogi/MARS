import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import train_cfg, model_cfg, path_cfg
from library.model import MultiScaleHierarchicalSKResNet
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    EarlyStopping,
    save_metrics,
    count_parameters,
)


class Trainer:
    """
    Manages the training and validation lifecycle of the Speech Command Recognition model.
    """

    def __init__(self, config=train_cfg, model_config=model_cfg, path_config=path_cfg):
        self.cfg = config
        self.model_cfg = model_config
        self.path_cfg = path_config
        self.device = torch.device(self.cfg.device)

        # Initialize Model
        self.model = MultiScaleHierarchicalSKResNet(self.model_cfg)
        self.model.to(self.device)

        print(f"Model initialized on {self.device}")
        print(f"Trainable parameters: {count_parameters(self.model)}")

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.T_max,
            eta_min=self.cfg.eta_min,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=self.cfg.patience,
            verbose=True,
            path=self.path_cfg.model_save_path,
            mode="max",  # We monitor validation accuracy
        )

        # Logging path
        self.log_path = os.path.join(self.path_cfg.working_dir, "training_log.csv")

    def train_epoch(self, train_loader, epoch):
        """Runs one epoch of training."""
        self.model.train()

        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")

        for i, (images, target) in enumerate(train_loader):
            images = images.to(self.device)
            target = target.to(self.device)

            # Forward pass
            output = self.model(images)
            loss = self.criterion(output, target)

            # Measure accuracy and record loss
            acc1 = calculate_accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses.avg, top1.avg

    def validate(self, val_loader):
        """Runs validation on the validation set."""
        self.model.eval()

        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")

        with torch.no_grad():
            for images, target in val_loader:
                images = images.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model(images)
                loss = self.criterion(output, target)

                # Measure accuracy and record loss
                acc1 = calculate_accuracy(output, target, topk=(1,))[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))

        return losses.avg, top1.avg

    def fit(self, train_loader, val_loader):
        """
        Main training loop.
        """
        print(f"Starting training for {self.cfg.epochs} epochs...")

        for epoch in range(1, self.cfg.epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Logging
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "time": epoch_time,
            }
            save_metrics(metrics, self.log_path)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc} | "
                f"LR: {current_lr} | Time: {epoch_time:.2f}s"
            )

            # Early Stopping Check
            self.early_stopping(val_acc, self.model, self.optimizer, epoch)

            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print("Training complete.")
        print(f"Best Validation Accuracy: {self.early_stopping.best_score}")
