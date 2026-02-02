import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


class Engine:
    """
    Handles the training, evaluation, and prediction loops for the DCNv2 model.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train_one_epoch(self, dataloader, optimizer, scheduler, criterion):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            continuous = batch["continuous"].to(self.device)
            categorical = batch["categorical"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(continuous, categorical)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Step the scheduler
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self, dataloader, criterion):
        """
        Evaluates the model on the validation set.
        Returns average loss and ROC AUC.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(continuous, categorical)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                all_targets.append(targets.cpu().numpy())
                all_preds.append(outputs.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate and flatten for metric calculation
        all_targets = np.concatenate(all_targets).ravel()
        all_preds = np.concatenate(all_preds).ravel()

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.0

        return avg_loss, auc

    def predict(self, dataloader):
        """
        Generates predictions for the test set.
        Returns IDs and predicted probabilities.
        """
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                ids = batch["id"]

                outputs = self.model(continuous, categorical)

                all_ids.extend(ids.numpy())
                all_preds.append(outputs.cpu().numpy().ravel())

        return np.array(all_ids), np.concatenate(all_preds)


def run_training(model, train_loader, val_loader, test_loader):
    """
    Orchestrates the training process, including:
    - Optimizer and Scheduler setup
    - Training loop with Early Stopping
    - Saving the best model
    - Generating submission file
    """
    device = torch.device(Config.DEVICE)
    print(f"Device selected: {device}")

    # Ensure working directory exists for model saving
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # Initialize Engine
    engine = Engine(model, device)

    # Loss Function (Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Early Stopping Variables
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = engine.train_one_epoch(
            train_loader, optimizer, scheduler, criterion
        )

        # Validate
        val_loss, val_auc = engine.evaluate(val_loader, criterion)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val AUC = {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # --- Inference ---
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    print("Generating predictions on test set...")
    ids, preds = engine.predict(test_loader)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {Config.ID_COL: ids.astype(int), Config.TARGET_COL: preds}
    )

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
