import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything, get_device
from library.models import OrthogonalSkipMLP


class MLPEngine:
    """
    Handles the training, validation, and inference of the OrthogonalSkipMLP model.
    """

    def __init__(self):
        """
        Initializes the engine, model, optimizer, and loss function.
        """
        seed_everything(Config.RANDOM_SEED)
        self.device = get_device()
        self.config = Config

        # Initialize Model
        self.model = OrthogonalSkipMLP().to(self.device)

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.MLP_LEARNING_RATE,
            weight_decay=self.config.MLP_WEIGHT_DECAY,
        )

        # Loss Function: BCEWithLogitsLoss (combines Sigmoid + BCELoss)
        self.criterion = nn.BCEWithLogitsLoss()

        # Path to save the best model
        self.best_model_path = os.path.join(self.config.CACHE_DIR, "best_mlp.pth")

    def train_one_epoch(self, dataloader):
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch)

            # Compute loss
            loss = self.criterion(logits, batch["target"])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns AUC score and probability predictions.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # Forward pass
                logits = self.model(batch)
                probs = torch.sigmoid(logits)

                # Store predictions and targets
                all_preds.extend(probs.cpu().numpy().flatten())
                all_targets.extend(batch["target"].cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate ROC AUC
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
        else:
            auc = 0.5  # Fallback if only one class is present in batch

        return auc, all_preds

    def predict(self, dataloader):
        """
        Generates predictions for a dataset (e.g., test set).
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # Forward pass
                logits = self.model(batch)
                probs = torch.sigmoid(logits)

                all_preds.extend(probs.cpu().numpy().flatten())

        return np.array(all_preds)

    def run(self, train_loader, val_loader, test_loader):
        """
        Executes the full training loop with early stopping.
        """
        print("--- Starting MLP Training Engine ---")
        best_auc = 0.0
        patience_counter = 0

        for epoch in range(self.config.MLP_NUM_EPOCHS):
            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_auc, _ = self.evaluate(val_loader)

            print(
                f"Epoch {epoch + 1}/{self.config.MLP_NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc}"
            )

            # Checkpointing & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.MLP_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        print(f"Training complete. Best Validation AUC: {best_auc}")

        # Load best model for final inference
        if os.path.exists(self.best_model_path):
            print("Loading best model checkpoint...")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: No best model found. Using current state.")

        # Generate final predictions
        print("Generating final predictions...")
        val_auc_final, val_probs = self.evaluate(val_loader)
        test_probs = self.predict(test_loader)

        return {
            "val_auc": val_auc_final,
            "val_probs": val_probs,
            "test_probs": test_probs,
        }
