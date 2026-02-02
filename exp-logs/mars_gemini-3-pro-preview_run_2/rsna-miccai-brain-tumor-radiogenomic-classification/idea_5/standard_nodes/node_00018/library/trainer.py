import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.model import MILEfficientNet
from library.data_loader import get_dataloaders

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class Trainer:
    """
    Trainer class for the Multi-Instance Attention-Pooled 2.5D Network.
    Handles training, validation, checkpointing, and submission generation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = MILEfficientNet().to(self.device)

        # Optimizer with aggressive weight decay as per design
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_auc = 0.0
        self.patience_counter = 0
        self.model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for inputs, targets in loader:
            # inputs: (B, C, H, W)
            # targets: (B,)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Collect predictions for AUC calculation
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)

        # Handle edge case where batch might have only one class
        if len(np.unique(all_targets)) < 2:
            epoch_auc = 0.5
        else:
            epoch_auc = roc_auc_score(all_targets, all_preds)

        return epoch_loss, epoch_auc

    def validate(self, loader):
        """
        Runs validation loop.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in loader:
                # inputs: (B, C, H, W)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)

        if len(np.unique(all_targets)) < 2:
            epoch_auc = 0.5
        else:
            epoch_auc = roc_auc_score(all_targets, all_preds)

        return epoch_loss, epoch_auc

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set using the best model.
        """
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.model_path):
            print(f"Loading best model from {self.model_path}")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            print("Warning: No best model found. Using current model weights.")

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                # inputs: (B, C, H, W)
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                # Flatten to 1D array
                all_probs.extend(probs.flatten())

        # Load test metadata to get IDs
        # The test loader iterates sequentially over the metadata, so order is preserved.
        df_test = pd.read_csv(Config.TEST_METADATA)

        # Ensure lengths match
        if len(all_probs) != len(df_test):
            print(
                f"Warning: Number of predictions ({len(all_probs)}) does not match number of test samples ({len(df_test)})."
            )

        # Assign predictions
        df_test["MGMT_value"] = all_probs

        # Format submission
        submission = df_test[["BraTS21ID", "MGMT_value"]]

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())

    def run(self):
        """
        Main execution method.
        """
        # Get DataLoaders
        print("Initializing DataLoaders...")
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

        print(f"Starting training on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_auc = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            print(
                f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss}, Train AUC: {train_auc}, "
                f"Val Loss: {val_loss}, Val AUC: {val_auc}"
            )

            # Checkpointing and Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved with AUC: {val_auc}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Generate submission after training
        self.generate_submission(test_loader)


def run_training():
    """
    Helper function to instantiate and run the trainer.
    """
    trainer = Trainer()
    trainer.run()
