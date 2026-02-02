import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    BEST_MODEL_PATH,
    SUBMISSION_FILE_PATH,
    BATCH_SIZE,
    LR,
    EPOCHS,
    PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    DEVICE,
    SEQ_LEN,
    NUM_TARGETS,
    SEED,
)
from library.utils import seed_everything, MCRMSELoss, metric_mcrmse
from library.data import get_dataloaders
from library.model import AHS_DFN


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = MCRMSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=True,
        )

        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for inputs, partner_indices, targets in self.train_loader:
            inputs = inputs.to(self.device)
            partner_indices = partner_indices.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass returns tuple (y_hat_1, y_hat_2)
            # y_hat_1: Pass 1 (Zero Feedback)
            # y_hat_2: Pass 2 (With Feedback)
            y_hat_1, y_hat_2 = self.model(inputs, partner_indices)

            # Calculate Loss
            # Anchored Loss: Loss(Pass2) + 0.5 * Loss(Pass1)
            # Both calculated over full sequence length (0-107) as per MCRMSELoss implementation
            loss_2 = self.criterion(y_hat_2, targets)
            loss_1 = self.criterion(y_hat_1, targets)
            loss = loss_2 + 0.5 * loss_1

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, partner_indices, targets in self.val_loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)

                # We use the final output (Pass 2) for validation
                _, y_hat_2 = self.model(inputs, partner_indices)

                all_preds.append(y_hat_2.cpu().numpy())
                all_targets.append(targets.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metric on scored subset only
        score = metric_mcrmse(all_preds, all_targets)
        return score

    def fit(self, epochs):
        print(f"Starting training on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_score = self.validate()

            self.scheduler.step(val_score)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score} | Time: {elapsed:.2f}s"
            )

            # Early Stopping and Model Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved (Score: {self.best_score})")
            else:
                self.patience_counter += 1
                print(f"  >>> Patience: {self.patience_counter}/{PATIENCE}")

            if self.patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_ids = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs, partner_indices, targets in test_loader:
                inputs = inputs.to(self.device)
                partner_indices = partner_indices.to(self.device)

                # Pass 2 is the final prediction
                _, y_hat_2 = self.model(inputs, partner_indices)

                all_preds.append(y_hat_2.cpu().numpy())
                # targets in test_loader are dummy, but we need IDs from dataset
                # The loader yields (inputs, pi, targets), we need to access IDs separately or assume order
                # The RNADataset stores IDs, but DataLoader collates.
                # Since we iterate sequentially with shuffle=False, we can retrieve IDs from dataset if needed,
                # but let's rely on the fact that we can't easily get IDs from the standard loop unless we modify collate or return them.
                # However, the provided library.data.RNADataset doesn't return IDs in __getitem__.
                # We will handle ID mapping in the formatting step by accessing the dataset directly or assuming order.

        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds


def generate_submission(preds, test_ids, output_path):
    """
    Formats predictions into the required CSV format.
    """
    print("Formatting submission...")

    # preds shape: (N_samples, 107, 5)
    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    data_rows = []

    for i, sample_id in enumerate(test_ids):
        sample_preds = preds[i]  # Shape (107, 5)

        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Construct row dict
            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            data_rows.append(row_dict)

    submission_df = pd.DataFrame(data_rows)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_and_predict(debug=False, epochs=EPOCHS):
    seed_everything(SEED)

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 2. Initialize Model
    print("Initializing AHS-DFN Model...")
    model = AHS_DFN().to(DEVICE)

    # 3. Train
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.fit(epochs)

    # 4. Predict
    preds = trainer.predict(test_loader)

    # 5. Generate Submission
    # Retrieve IDs from the test dataset
    test_ids = test_loader.dataset.ids
    generate_submission(preds, test_ids, SUBMISSION_FILE_PATH)
