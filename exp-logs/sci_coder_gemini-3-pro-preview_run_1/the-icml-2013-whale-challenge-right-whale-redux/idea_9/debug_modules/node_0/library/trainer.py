import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model

# Ensure reproducibility
utils.set_seed(config.SEED)


class Trainer:
    def __init__(self, load_cached_data=True, debug=False):
        """
        Initializes the Trainer class.

        Args:
            load_cached_data (bool): Whether to load pre-processed data from cache.
            debug (bool): Whether to run in debug mode with a smaller dataset.
        """
        self.device = config.DEVICE
        self.debug = debug

        # Initialize Model
        print("Initializing model...")
        self.model = model.SKResNetCRNN().to(self.device)

        # Loss Function
        # handling class imbalance with pos_weight
        pos_weight = torch.tensor([config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Scheduler
        # Mode is 'max' because we want to maximize AUC
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # DataLoaders
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = dataset.get_dataloaders(
            load_cached_data=load_cached_data, debug=debug
        )

        # Checkpoint path
        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Apply Mixup
            data, target_a, target_b, lam = dataset.mixup_data(
                data, target, alpha=config.MIXUP_ALPHA, use_cuda=True
            )

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Compute Loss (Mixup criterion)
            # BCEWithLogitsLoss expects float targets for mixup
            # target is (Batch,) -> unsqueeze to (Batch, 1) for BCE
            target_a = target_a.unsqueeze(1)
            target_b = target_b.unsqueeze(1)

            loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(
                output, target_b
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # Loss calculation (standard)
                target_unsqueezed = target.unsqueeze(1)
                loss = self.criterion(output, target_unsqueezed)
                running_loss += loss.item()

                # Store predictions (sigmoid) and targets for AUC
                preds = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = running_loss / len(self.val_loader)

        # Compute AUC
        # Flatten list of arrays
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        auc_score = utils.compute_score(all_targets, all_preds)

        return avg_loss, auc_score

    def fit(self, epochs=config.EPOCHS, patience=5):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        best_auc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_auc = self.validate()

            # Scheduler Step
            self.scheduler.step(val_auc)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Val AUC: {best_auc}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        """
        print("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            print(
                "No best model found. Using current model weights (warning: might be suboptimal)."
            )
        else:
            print(f"Loading weights from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)

                output = self.model(data)
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                predictions.extend(probs)

        # Create Submission DataFrame
        # We need the clip names from the test metadata
        test_df = pd.read_csv(config.TEST_METADATA_PATH)

        if self.debug:
            test_df = test_df.iloc[: config.DEBUG_SIZE]

        # Ensure lengths match
        if len(predictions) != len(test_df):
            print(
                f"Warning: Number of predictions ({len(predictions)}) does not match number of test files ({len(test_df)})."
            )
            # Truncate or pad if necessary (though shouldn't happen with correct loaders)
            if len(predictions) > len(test_df):
                predictions = predictions[: len(test_df)]
            else:
                # Pad with zeros
                predictions.extend([0.0] * (len(test_df) - len(predictions)))

        submission_df = pd.DataFrame(
            {"clip": test_df["clip"], "probability": predictions}
        )

        # Save submission
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

        return submission_df
