import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import ResGLUNet


class Trainer:
    """
    Encapsulates the training, evaluation, and prediction logic for the ResGLUNet model.
    """

    def __init__(self, model, device=None):
        """
        Args:
            model (nn.Module): The ResGLUNet model to train.
            device (torch.device, optional): The compute device. Defaults to Config.DEVICE.
        """
        self.model = model
        self.device = device if device else torch.device(Config.DEVICE)
        self.model.to(self.device)

        # Optimizer: AdamW with high weight decay as per Idea description
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss: Binary Cross Entropy (Model outputs probabilities via Sigmoid)
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            x_cat = batch["cat"].to(self.device)
            x_cont = batch["cont"].to(self.device)
            y = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x_cat, x_cont)
            loss = self.criterion(preds, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns:
            Tuple[float, float]: (Average Loss, AUC Score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x_cat = batch["cat"].to(self.device)
                x_cont = batch["cont"].to(self.device)
                y = batch["target"].to(self.device)

                preds = self.model(x_cat, x_cont)
                loss = self.criterion(preds, y)

                running_loss += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = running_loss / len(val_loader)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Compute AUC
        try:
            auc_score = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # Handle edge case where only one class is present in validation batch
            auc_score = 0.5

        return avg_loss, auc_score

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Runs the full training loop with early stopping.
        """
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.evaluate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        # Load the best model weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                x_cat = batch["cat"].to(self.device)
                x_cont = batch["cont"].to(self.device)

                preds = self.model(x_cat, x_cont)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds).flatten()


def run_experiment(debug_samples=None):
    """
    Orchestrates the training and submission generation process.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_samples=debug_samples
    )

    # 3. Model Initialization
    model = ResGLUNet()

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # 5. Inference
    print("Generating predictions...")
    predictions = trainer.predict(test_loader)

    # 6. Submission
    test_meta = pd.read_csv(Config.TEST_METADATA)
    submission = pd.DataFrame({"id": test_meta["id"], "target": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
