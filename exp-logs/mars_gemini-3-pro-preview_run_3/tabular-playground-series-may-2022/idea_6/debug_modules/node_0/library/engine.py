import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_device


class Engine:
    """
    Encapsulates training, evaluation, and inference logic for the Hybrid Transformer-Funnel model.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model (nn.Module): The neural network model to train/evaluate.
            device (torch.device, optional): The device to run on. Defaults to auto-detection.
        """
        self.model = model
        self.device = device if device else get_device()
        self.model.to(self.device)

    def train_one_epoch(self, dataloader, optimizer, scheduler, criterion):
        """
        Trains the model for one epoch.

        Args:
            dataloader (DataLoader): Training data loader.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): PyTorch learning rate scheduler.
            criterion (Loss): Loss function.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            # Unpack batch and move to device
            cat_seq = batch["cat_seq"].to(self.device)
            cont_vec = batch["cont_vec"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(cat_seq, cont_vec)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Scheduler step (OneCycleLR steps per batch)
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def evaluate(self, dataloader, criterion):
        """
        Evaluates the model on the validation set.

        Args:
            dataloader (DataLoader): Validation data loader.
            criterion (Loss): Loss function.

        Returns:
            tuple: (average_loss, roc_auc_score)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                cat_seq = batch["cat_seq"].to(self.device)
                cont_vec = batch["cont_vec"].to(self.device)
                targets = batch["target"].to(self.device)

                logits = self.model(cat_seq, cont_vec)
                loss = criterion(logits, targets)

                total_loss += loss.item()

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Calculate AUC
        auc = roc_auc_score(all_targets, all_preds)
        avg_loss = total_loss / num_batches

        return avg_loss, auc

    def fit(self, train_loader, val_loader):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        """
        # Hyperparameters from Config
        epochs = Config.EPOCHS
        lr = Config.LEARNING_RATE
        weight_decay = Config.WEIGHT_DECAY
        patience = Config.PATIENCE
        save_path = Config.MODEL_SAVE_PATH

        # Setup components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # OneCycleLR Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Standard setting for OneCycle
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_one_epoch(
                train_loader, optimizer, scheduler, criterion
            )

            # Validate
            val_loss, val_auc = self.evaluate(val_loader, criterion)

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val AUC: {val_auc:.10f}"
            )

            # Early Stopping and Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with AUC: {best_auc:.10f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

        # Load best model for future use (e.g. inference immediately after fit)
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            np.ndarray: Array of predicted probabilities.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                cat_seq = batch["cat_seq"].to(self.device)
                cont_vec = batch["cont_vec"].to(self.device)

                logits = self.model(cat_seq, cont_vec)
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds).flatten()

    def generate_submission(self, test_loader, test_ids):
        """
        Generates the submission file.

        Args:
            test_loader (DataLoader): Test data loader.
            test_ids (np.ndarray or list): IDs corresponding to the test set.
        """
        print("Generating predictions for submission...")
        predictions = self.predict(test_loader)

        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(test_ids)} IDs"
            )

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
        )

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
