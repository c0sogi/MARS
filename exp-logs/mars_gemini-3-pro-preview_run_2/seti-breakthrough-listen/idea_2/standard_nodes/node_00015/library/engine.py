import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from library.config import Config
from library.utils import get_score, seed_everything


class SETIEngine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_auc = 0.0
        self.patience = 5
        self.counter = 0

        self.scaler = amp.GradScaler()

        # Ensure reproducibility
        seed_everything(Config.seed)

    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        self.optimizer.zero_grad()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            batch_size = images.size(0)

            with amp.autocast():
                if Config.use_mixup and Config.mixup_alpha > 0:
                    images, targets_a, targets_b, lam = self.mixup_data(
                        images, targets, Config.mixup_alpha
                    )
                    outputs = self.model(images)
                    loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                # Normalize loss for gradient accumulation
                loss = loss / Config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % Config.grad_accum_steps == 0 or (
                step + 1 == len(train_loader)
            ):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Restore loss value for logging
            running_loss += loss.item() * Config.grad_accum_steps * batch_size
            dataset_size += batch_size

            if step % Config.print_freq == 0:
                print(
                    f"Epoch: {epoch} Step: {step} Train Loss: {loss.item() * Config.grad_accum_steps}"
                )

        epoch_loss = running_loss / dataset_size
        print(f"Epoch: {epoch} Training Loss: {epoch_loss}")
        return epoch_loss

    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        preds = []
        valid_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                batch_size = images.size(0)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                preds.append(torch.sigmoid(outputs).cpu().numpy())
                valid_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / dataset_size
        preds = np.concatenate(preds)
        valid_targets = np.concatenate(valid_targets)
        auc = get_score(valid_targets, preds)

        # Print full precision as requested
        print(f"Epoch: {epoch} Validation Loss: {epoch_loss} AUC: {auc}")
        return epoch_loss, auc

    def train(self, train_loader, val_loader, epochs):
        Config.create_dirs()

        for epoch in range(epochs):
            self.train_one_epoch(train_loader, epoch)
            val_loss, val_auc = self.validate_one_epoch(val_loader, epoch)

            if self.scheduler:
                self.scheduler.step()

            # Checkpoint and Early Stopping
            if val_auc > self.best_auc:
                print(f"Validation AUC improved from {self.best_auc} to {val_auc}")
                self.best_auc = val_auc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.output_dir, "best_model.pth"),
                )
                self.counter = 0
            else:
                self.counter += 1
                print(f"No improvement in AUC. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print("Early stopping triggered")
                break

    def predict(self, test_loader):
        best_model_path = os.path.join(Config.output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print(f"Loaded model from {best_model_path}")
        else:
            print("Warning: Best model not found, using current model weights.")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                with amp.autocast():
                    # TTA: Original
                    out1 = torch.sigmoid(self.model(images))

                    # TTA: Horizontal Flip (Frequency Axis - dim 3)
                    # Input shape (B, C, H, W). W is frequency.
                    images_flipped = torch.flip(images, dims=[3])
                    out2 = torch.sigmoid(self.model(images_flipped))

                # Average predictions
                avg_preds = (out1 + out2) / 2.0
                preds.append(avg_preds.float().cpu().numpy())

        predictions = np.concatenate(preds).flatten()

        # Generate Submission
        # Assumes test_loader is not shuffled and matches test_df order
        test_df = test_loader.dataset.df
        submission = pd.DataFrame({"id": test_df["id"], "target": predictions})

        submission_path = os.path.join(Config.submission_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
