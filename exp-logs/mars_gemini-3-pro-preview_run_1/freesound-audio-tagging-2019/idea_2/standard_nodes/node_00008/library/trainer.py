import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import CFG
from library.utils import set_seed, calculate_overall_lwlrap
from library.dataset import get_dataloader
from library.model import AudioEfficientNet


# --- Mixup Utilities ---
def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device != "cpu":
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    def __init__(self, load_cached_data=False):
        """
        Args:
            load_cached_data (bool): Placeholder for caching logic requirement.
                                     Since data loading is on-the-fly, this is not actively used
                                     but kept for interface consistency.
        """
        set_seed(CFG.seed)
        self.device = CFG.device
        self.output_dir = CFG.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Model
        self.model = AudioEfficientNet(
            model_name=CFG.model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        # Criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            # Apply Mixup
            if np.random.rand() < CFG.mixup_prob:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, CFG.mixup_alpha, self.device
                )

                outputs = self.model(images)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(loader.dataset)

    def validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                # Apply sigmoid for metric calculation
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate LWLRAP
        lwlrap = calculate_overall_lwlrap(all_targets, all_preds)
        avg_loss = running_loss / len(loader.dataset)

        return avg_loss, lwlrap

    def fit(self, epochs=CFG.epochs, patience=7):
        print(f"Starting training for {epochs} epochs on device {self.device}...")

        train_loader = get_dataloader("train", debug=CFG.debug)
        val_loader = get_dataloader("val", debug=CFG.debug)

        best_score = -np.inf
        patience_counter = 0
        best_model_path = os.path.join(self.output_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate_epoch(val_loader)

            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val LWLRAP: {val_score} - "
                f"Time: {elapsed:.2f}s"
            )

            # Save Best Model
            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val LWLRAP: {best_score}")

    def predict(self):
        print("Starting inference on test set...")

        # Load Best Model
        best_model_path = os.path.join(self.output_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at {best_model_path}")

        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)

        test_loader = get_dataloader("test", debug=CFG.debug)

        all_preds = []
        all_fnames = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                fnames = batch["fname"]

                outputs = self.model(images)
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_fnames.extend(fnames)

        all_preds = np.concatenate(all_preds, axis=0)

        # Create Submission DataFrame
        # Retrieve label columns from the dataset to ensure correct order
        label_cols = test_loader.dataset.label_cols

        submission_df = pd.DataFrame(all_preds, columns=label_cols)
        submission_df.insert(0, "fname", all_fnames)

        # Ensure output directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


def run_training():
    trainer = Trainer()
    trainer.fit()
    trainer.predict()
