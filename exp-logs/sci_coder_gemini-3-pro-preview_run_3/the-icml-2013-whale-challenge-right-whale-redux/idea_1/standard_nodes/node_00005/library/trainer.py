import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_dataloaders
from library.model import ShallowCNN


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelTrainer:
    """
    Manages the training lifecycle, validation, and submission generation
    for the Right Whale Detection task.
    """

    def __init__(self, epochs=Config.EPOCHS, debug=Config.DEBUG):
        """
        Initialize the trainer with hyperparameters and load data.

        Args:
            epochs (int): Number of training epochs.
            debug (bool): If True, uses a small subset of data for debugging.
        """
        # Update Config debug flag if necessary before loading data
        if debug != Config.DEBUG:
            Config.DEBUG = debug

        self.epochs = epochs

        # Ensure reproducibility
        set_seed(Config.SEED)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing ModelTrainer on device: {self.device}")

        # Initialize Model
        self.model = ShallowCNN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        # Loss Function: BCELoss with manual weighting handled in the loop
        self.criterion = nn.BCELoss(reduction="none")

        # Load Data
        print("Loading dataloaders...")
        self.dataloaders = get_dataloaders()

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()

            # Apply Mixup if enabled
            if hasattr(Config, "MIXUP_ALPHA") and Config.MIXUP_ALPHA > 0:
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
                index = torch.randperm(inputs.size(0)).to(self.device)

                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                outputs = self.model(mixed_inputs)

                # Helper to calculate weighted loss
                def calc_weighted_loss(pred, target):
                    loss_unreduced = self.criterion(pred, target)
                    weights = target * Config.POS_WEIGHT + (1 - target)
                    return (loss_unreduced * weights).mean()

                loss_a = calc_weighted_loss(outputs, labels)
                loss_b = calc_weighted_loss(outputs, labels[index])
                loss = lam * loss_a + (1 - lam) * loss_b

            else:
                outputs = self.model(inputs)
                # Calculate weighted loss to handle class imbalance
                loss_unreduced = self.criterion(outputs, labels)
                # Weight: POS_WEIGHT for label 1, 1.0 for label 0
                weights = labels * Config.POS_WEIGHT + (1 - labels)
                loss = (loss_unreduced * weights).mean()

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        return running_loss / total_samples

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and ROC AUC score.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                outputs = self.model(inputs)

                # Validation loss calculation
                loss_unreduced = self.criterion(outputs, labels)
                weights = labels * Config.POS_WEIGHT + (1 - labels)
                loss = (loss_unreduced * weights).mean()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = running_loss / total_samples

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # Fallback if only one class is present in batch/subset
            auc = 0.5

        return avg_loss, auc

    def train(self):
        """
        Executes the full training loop with Early Stopping.
        Returns the path to the best saved model.
        """
        best_val_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        train_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["val"]

        print(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Step the scheduler
            self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
                    )
                    break

        print(f"Training finished. Best model saved to {best_model_path}")
        return best_model_path

    def generate_submission(self, model_path):
        """
        Loads the best model and generates predictions for the test set.
        Saves the result to submission.csv.
        """
        print(f"Generating submission using model at {model_path}...")

        # Load best model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        test_loader = self.dataloaders["test"]
        predictions = []
        clip_names_list = []

        with torch.no_grad():
            for inputs, clip_names in test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)

                # Flatten predictions to 1D array
                probs = outputs.cpu().numpy().flatten()

                predictions.extend(probs)
                clip_names_list.extend(clip_names)

        # Create Submission DataFrame
        df = pd.DataFrame({"clip": clip_names_list, "probability": predictions})

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main entry point to run the training pipeline.
    """
    trainer = ModelTrainer()
    best_model_path = trainer.train()
    trainer.generate_submission(best_model_path)
