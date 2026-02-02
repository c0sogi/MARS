import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import DataProcessor, ManufacturingDataset
from library.model import ARPFEModel


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the ARPFE model.
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initialize the Trainer. Sets up device, seeds, data, and model.

        Args:
            load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
        """
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        print(f"Initializing Trainer on device: {self.device}")

        # 1. Load and Process Data
        self.train_df, self.val_df, self.test_df, self.vocab_sizes = (
            DataProcessor.process_data(load_cached_data=load_cached_data)
        )

        # 2. Initialize Datasets
        self.train_dataset = ManufacturingDataset(self.train_df, is_test=False)
        self.val_dataset = ManufacturingDataset(self.val_df, is_test=False)
        self.test_dataset = ManufacturingDataset(self.test_df, is_test=True)

        # 3. Initialize Dataloaders
        # Using 4 workers for efficient data loading
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # 4. Initialize Model
        self.model = ARPFEModel(vocab_sizes=self.vocab_sizes).to(self.device)

        # 5. Loss Function
        # We use BCEWithLogitsLoss. The reduction will be handled manually to sum across streams.
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, epochs: int = Config.EPOCHS, patience: int = 5):
        """
        Executes the training loop with Early Stopping and OneCycleLR scheduler.

        Args:
            epochs (int): Maximum number of training epochs.
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        print("Starting training...")

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        steps_per_epoch = len(self.train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        best_auc = -float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for cont_x, cat_x, targets in self.train_loader:
                cont_x = cont_x.to(self.device)
                cat_x = cat_x.to(self.device)
                targets = targets.to(self.device)  # Shape (Batch, 1)

                optimizer.zero_grad()

                # Forward Pass -> Returns (Batch, 5)
                outputs = self.model(cont_x, cat_x)

                # Compute Loss
                # Sum of BCE losses for each of the 5 independent streams
                loss = 0
                for i in range(5):
                    # Select stream i output: (Batch, ) -> View as (Batch, 1)
                    stream_out = outputs[:, i].view(-1, 1)
                    loss += self.criterion(stream_out, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)

            # Validation Step
            val_auc, val_loss = self.validate()

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc}"
                    )
                    break

        # Load best model for future use
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}...")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: (auc_score, average_loss)
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for cont_x, cat_x, targets in self.val_loader:
                cont_x = cont_x.to(self.device)
                cat_x = cat_x.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(cont_x, cat_x)

                # Calculate Loss
                loss = 0
                for i in range(5):
                    stream_out = outputs[:, i].view(-1, 1)
                    loss += self.criterion(stream_out, targets)
                running_loss += loss.item()

                # Calculate Predictions
                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                # Arithmetic mean across the 5 streams
                avg_probs = probs.mean(dim=1)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(avg_probs.cpu().numpy())

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        val_loss = running_loss / len(self.val_loader)
        val_auc = compute_auc(all_targets, all_preds)

        return val_auc, val_loss

    def generate_submission(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Generating submission...")
        self.model.eval()
        all_preds = []

        # Retrieve IDs from dataset
        ids = self.test_dataset.ids

        with torch.no_grad():
            for cont_x, cat_x in self.test_loader:
                cont_x = cont_x.to(self.device)
                cat_x = cat_x.to(self.device)

                outputs = self.model(cont_x, cat_x)

                # Arithmetic mean of probabilities
                probs = torch.sigmoid(outputs)
                avg_probs = probs.mean(dim=1)

                all_preds.append(avg_probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": ids, "target": all_preds})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
