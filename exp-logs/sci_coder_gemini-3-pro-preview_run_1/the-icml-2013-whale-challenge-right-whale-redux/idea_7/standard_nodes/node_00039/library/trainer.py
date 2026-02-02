import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, calculate_auc
from library.data_loader import get_dataloaders
from library.model import MultiResResNet34CRNN


class Trainer:
    """
    Manages the training, validation, and evaluation of the model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.best_auc = 0.0

    def train_epoch(self):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for inputs, targets_a, targets_b, lam in self.train_loader:
            inputs = inputs.to(self.device)
            targets_a = targets_a.to(self.device).view(-1, 1)
            targets_b = targets_b.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Mixup Loss Calculation
            # Loss is weighted average of loss against target_a and target_b
            loss = lam * self.criterion(outputs, targets_a) + (
                1 - lam
            ) * self.criterion(outputs, targets_b)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / n_batches

    def validate_epoch(self):
        """
        Runs validation on unmixed data and calculates AUC.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        avg_loss = running_loss / n_batches
        auc = calculate_auc(all_targets, all_preds)
        return avg_loss, auc

    def fit(self, epochs, patience=5):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate_epoch()

            # Step the scheduler based on Validation AUC
            self.scheduler.step(val_auc)

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Time: {duration:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with AUC: {self.best_auc}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        print(f"Training finished. Best Validation AUC: {self.best_auc}")
        return self.best_auc


def train_model(debug=False, epochs=Config.NUM_EPOCHS, load_cached_data=True):
    """
    Sets up the environment, loads data, and runs the training process.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # Caching is handled internally by get_dataloaders -> process_dataset
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug_subset=debug
    )

    # 2. Initialize Model
    # Use pretrained ImageNet weights for the backbone
    model = MultiResResNet34CRNN(pretrained=True).to(device)

    # 3. Setup Loss, Optimizer, Scheduler
    # Explicit positive class weight to handle imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Reduce LR when AUC plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # 4. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=save_path,
    )

    trainer.fit(epochs=epochs, patience=5)
    return save_path


def predict_and_submit(model_path, debug=False, load_cached_data=True):
    """
    Loads the best model, predicts on the test set, and saves the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Test Data
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug_subset=debug
    )

    # 2. Load Model
    # Initialize structure without downloading weights, then load state dict
    model = MultiResResNet34CRNN(pretrained=False).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Generate Predictions
    all_preds = []
    print("Generating predictions on test set...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy().flatten())

    # 4. Create Submission File
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        df_test = df_test.iloc[: len(all_preds)]

    # Safety check for length mismatch
    if len(all_preds) != len(df_test):
        print(
            f"Warning: Prediction count {len(all_preds)} does not match test set size {len(df_test)}."
        )
        if len(all_preds) > len(df_test):
            all_preds = all_preds[: len(df_test)]
        else:
            all_preds.extend([0.0] * (len(df_test) - len(all_preds)))

    df_test["probability"] = all_preds

    # Save to submission directory
    submission_df = df_test[["clip", "probability"]]
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
