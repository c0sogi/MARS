import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.utils import calculate_roc_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, fold, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.fold = fold
        self.model_name = model_name
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # Scheduler (Monotonic Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
        )

        # Loss Function
        # Using BCEWithLogitsLoss as specialized imbalance losses can suppress gradients
        # too much on small datasets (Cite solution_lesson_node_00034)
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Best Score Tracking
        self.best_auc = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Apply Mixup
            use_mixup = self.cfg.mixup_alpha > 0
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.cfg.mixup_alpha, self.device
                )

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images)
                if use_mixup:
                    loss = mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    loss = self.criterion(outputs, labels)

            # Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Apply sigmoid to logits to get probabilities
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.detach().cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate AUC
        auc = calculate_roc_auc(all_targets, all_preds)
        avg_loss = running_loss / len(self.val_loader)

        return auc, avg_loss

    def save_checkpoint(self, auc):
        filename = f"{self.model_name}_fold_{self.fold}_best.pth"
        save_path = os.path.join(self.cfg.working_dir, filename)

        torch.save(
            {
                "fold": self.fold,
                "model_state_dict": self.model.state_dict(),
                "best_auc": float(auc),
                "config": self.cfg.__dict__,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")

    def fit(self):
        print(f"Starting training for {self.model_name} - Fold {self.fold}")

        for epoch in range(self.cfg.epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_auc, val_loss = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            end_time = time.time()
            epoch_time = end_time - start_time

            print(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc} | "
                f"LR: {current_lr} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Save Best Model
            if val_auc > self.best_auc:
                print(f"Validation AUC improved from {self.best_auc} to {val_auc}")
                self.best_auc = val_auc
                self.save_checkpoint(val_auc)

        print(f"Training finished. Best AUC: {self.best_auc}")
        return self.best_auc


def run_training(cfg, model, train_loader, val_loader, fold, model_name):
    """
    Helper function to instantiate Trainer and run the training loop.
    """
    trainer = Trainer(model, train_loader, val_loader, cfg, fold, model_name)
    best_score = trainer.fit()
    return best_score
