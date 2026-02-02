import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.dataset import EEGMultiModalDataset
from library.model import DualStreamEfficientNet
from library.transforms import mixup_criterion


class Engine:
    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        seed_everything(config.SEED)

        # Ensure output directories exist
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.SUBMISSION_FILE), exist_ok=True)

    def train_one_epoch(self, model, loader, optimizer, scheduler, scaler, epoch):
        model.train()
        running_loss = 0.0
        dataset_size = 0

        # KLDivLoss expects input to be log-probabilities
        criterion = nn.KLDivLoss(reduction="batchmean")

        for batch_idx, data in enumerate(loader):
            eeg_spec = data["eeg_spec"].to(self.device, non_blocking=True)
            kaggle_spec = data["kaggle_spec"].to(self.device, non_blocking=True)
            targets = data["target"].to(self.device, non_blocking=True)

            batch_size = eeg_spec.size(0)

            optimizer.zero_grad()

            with autocast(enabled=self.config.USE_AMP):
                # Apply MixUp manually to handle dual streams consistently
                if self.config.MIXUP_ALPHA > 0:
                    lam = np.random.beta(
                        self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA
                    )
                    index = torch.randperm(batch_size).to(self.device)

                    # Mix both input streams using the same lambda and indices
                    mixed_eeg = lam * eeg_spec + (1 - lam) * eeg_spec[index, :]
                    mixed_kaggle = lam * kaggle_spec + (1 - lam) * kaggle_spec[index, :]

                    # Mix targets
                    y_a, y_b = targets, targets[index]

                    # Forward pass
                    logits = model(mixed_eeg, mixed_kaggle)
                    log_probs = F.log_softmax(logits, dim=1)

                    # Compute MixUp loss
                    loss = mixup_criterion(criterion, log_probs, y_a, y_b, lam)

                else:
                    # Standard forward pass
                    logits = model(eeg_spec, kaggle_spec)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, targets)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def validate(self, model, loader):
        model.eval()
        running_loss = 0.0
        dataset_size = 0

        # Store predictions and targets for full metric calculation
        all_preds = []
        all_targets = []

        criterion = nn.KLDivLoss(reduction="batchmean")

        with torch.no_grad():
            for data in loader:
                eeg_spec = data["eeg_spec"].to(self.device, non_blocking=True)
                kaggle_spec = data["kaggle_spec"].to(self.device, non_blocking=True)
                targets = data["target"].to(self.device, non_blocking=True)

                batch_size = eeg_spec.size(0)

                with autocast(enabled=self.config.USE_AMP):
                    logits = model(eeg_spec, kaggle_spec)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Convert logits to probabilities for metric calculation
                probs = F.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / dataset_size

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metric using provided utility
        metric_score = kl_divergence_score(all_targets, all_preds)

        return avg_loss, metric_score

    def run_training(self):
        print(
            f"Loading metadata from {self.config.TRAIN_CSV} and {self.config.VAL_CSV}"
        )
        df_train = pd.read_csv(self.config.TRAIN_CSV)
        df_val = pd.read_csv(self.config.VAL_CSV)

        if self.config.DEBUG:
            print("Debug mode enabled: using subset of data.")
            df_train = df_train.head(100)
            df_val = df_val.head(100)

        # Initialize Datasets
        train_dataset = EEGMultiModalDataset(df_train, self.config, mode="train")
        val_dataset = EEGMultiModalDataset(df_val, self.config, mode="val")

        # Initialize DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Initialize Model
        model = DualStreamEfficientNet(self.config)
        model.to(self.device)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY
        )

        # Cosine Annealing Scheduler updated every step
        num_train_steps = len(train_loader) * self.config.EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=1e-6)

        scaler = GradScaler(enabled=self.config.USE_AMP)

        best_score = float("inf")
        best_model_path = os.path.join(self.config.OUTPUT_DIR, "best_model.pth")

        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, epoch
            )

            val_loss, val_score = self.validate(model, val_loader)

            print(f"Epoch {epoch}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Metric (KL): {val_score}")

            # Save Best Model
            if val_score < best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved Best Model! Score: {best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

        return best_model_path

    def generate_submission(self, model_path):
        print(f"Loading test metadata from {self.config.TEST_CSV}")
        df_test = pd.read_csv(self.config.TEST_CSV)

        test_dataset = EEGMultiModalDataset(df_test, self.config, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Load Model
        model = DualStreamEfficientNet(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        all_probs = []

        print("Generating predictions on test set...")

        with torch.no_grad():
            for data in test_loader:
                eeg_spec = data["eeg_spec"].to(self.device)
                kaggle_spec = data["kaggle_spec"].to(self.device)

                with autocast(enabled=self.config.USE_AMP):
                    logits = model(eeg_spec, kaggle_spec)
                    probs = F.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(all_probs, columns=self.config.CLASS_NAMES)
        submission_df.insert(0, "eeg_id", df_test["eeg_id"])

        # Save
        submission_df.to_csv(self.config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_FILE}")
