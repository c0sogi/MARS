import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ToxicityDataset
from library.model import ToxicityModel
from library.utils import seed_everything, get_score


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    """

    def __init__(self):
        self.device = Config.device
        self.best_score = -np.inf

        # Ensure working and submission directories exist
        os.makedirs(Config.working_dir, exist_ok=True)
        os.makedirs(Config.submission_dir, exist_ok=True)

        self.model_path = os.path.join(Config.working_dir, "model.pth")

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, criterion):
        """
        Performs one epoch of training.
        """
        model.train()
        running_loss = 0.0
        dataset_size = 0

        for step, data in enumerate(dataloader):
            ids = data["input_ids"].to(self.device, dtype=torch.long)
            mask = data["attention_mask"].to(self.device, dtype=torch.long)
            targets = data["labels"].to(self.device, dtype=torch.float)

            batch_size = ids.size(0)

            # Forward pass
            outputs = model(ids, mask)

            # Loss calculation
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if Config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimization step
            optimizer.step()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def valid_one_epoch(self, model, dataloader, criterion):
        """
        Performs one epoch of validation.
        """
        model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data in dataloader:
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                targets = data["labels"].to(self.device, dtype=torch.float)

                batch_size = ids.size(0)

                outputs = model(ids, mask)
                loss = criterion(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Store targets and predictions (apply sigmoid for probabilities)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        epoch_loss = running_loss / dataset_size
        epoch_score = get_score(all_targets, all_preds)

        return epoch_loss, epoch_score

    def fit(self, load_cached_data=True):
        """
        Main training loop with early stopping.
        """
        seed_everything(Config.seed)

        print("Initializing Datasets...")
        train_dataset = ToxicityDataset(
            split="train", load_cached_data=load_cached_data
        )
        val_dataset = ToxicityDataset(split="val", load_cached_data=load_cached_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
        )

        print("Initializing Model...")
        model = ToxicityModel()
        model.to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Scheduler (OneCycleLR)
        # Calculate total steps based on dataset size and epochs
        steps_per_epoch = len(train_loader)
        num_train_steps = steps_per_epoch * Config.epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.learning_rate,
            total_steps=num_train_steps,
            pct_start=Config.pct_start,
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        patience = 0
        early_stopping_patience = 3

        print(f"Starting training for {Config.epochs} epochs on {self.device}...")

        for epoch in range(Config.epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            val_loss, val_score = self.valid_one_epoch(model, val_loader, criterion)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.2f}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val AUC: {val_score}")

            # Early Stopping and Model Saving
            if val_score > self.best_score:
                print(
                    f"Validation Score Improved ({self.best_score} -> {val_score}). Saving model..."
                )
                self.best_score = val_score
                torch.save(model.state_dict(), self.model_path)
                patience = 0
            else:
                patience += 1
                print(f"No improvement. Patience: {patience}/{early_stopping_patience}")

            if patience >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val AUC: {self.best_score}")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Starting Inference...")

        # Load Test Data
        test_dataset = ToxicityDataset(split="test", load_cached_data=load_cached_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
        )

        # Load Model
        model = ToxicityModel()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first."
            )

        print(f"Loading model weights from {self.model_path}...")
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        all_preds = []

        # Inference Loop
        with torch.no_grad():
            for data in test_loader:
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)

                outputs = model(ids, mask)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)

        final_preds = np.concatenate(all_preds)

        # Create Submission DataFrame
        # Load test metadata to get the correct IDs
        test_meta = pd.read_csv(Config.test_meta_path)

        # If in debug mode, the dataset only processed a subset, so we must slice the metadata
        if Config.debug:
            test_meta = test_meta.iloc[: Config.debug_subset_size]

        submission = pd.DataFrame(final_preds, columns=Config.target_cols)
        submission.insert(0, "id", test_meta["id"])

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
