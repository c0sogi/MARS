import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss, MetricTracker
from library.data_loader import prepare_data
from library.model import MCRHNet


class Runner:
    """
    Manages the training, validation, and submission generation for MCRH-Net.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        # Load Data
        print("Initializing Runner and loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.input_shape = (
            prepare_data(load_cached_data=True)
        )

        # Initialize Model
        print(f"Initializing MCRHNet with input shape: {self.input_shape}")
        self.model = MCRHNet(input_dim=self.input_shape).to(self.device)

        # Loss Function
        self.criterion = MaskedL1Loss().to(self.device)

        # Optimizer (AdamW with low weight decay as per Idea 25)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (OneCycleLR for 80 epochs)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # State tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        tracker = MetricTracker()

        for batch_idx, (x, y, u_out) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            u_out = u_out.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Calculate Masked L1 Loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimization step
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            tracker.update(loss.item(), x.size(0))

        return tracker.avg

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for x, y, u_out in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                u_out = u_out.to(self.device)

                preds = self.model(x)
                loss = self.criterion(preds, y, u_out)

                tracker.update(loss.item(), x.size(0))

        return tracker.avg

    def train(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpointing
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation Loss: {self.best_val_loss}")

    def generate_submission(self):
        """
        Generates submission file using the best trained model.
        """
        print("Generating submission...")

        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for x, u_out in self.test_loader:
                x = x.to(self.device)
                # u_out is not used for prediction, only for loss masking during training

                # Forward pass
                preds = self.model(x)

                # Flatten predictions: (Batch, Seq, 1) -> (Batch * Seq)
                preds_flat = preds.view(-1).cpu().numpy()
                predictions.append(preds_flat)

        # Concatenate all batches
        all_predictions = np.concatenate(predictions)

        # Load Test IDs
        # prepare_data saves test_ids.npy in the working directory
        test_ids_path = Config.CACHE_TEST_IDS
        if not os.path.exists(test_ids_path):
            raise FileNotFoundError(f"Test IDs file not found at {test_ids_path}")

        test_ids = np.load(test_ids_path)

        # Ensure lengths match
        if len(test_ids) != len(all_predictions):
            print(
                f"Warning: Length mismatch. IDs: {len(test_ids)}, Preds: {len(all_predictions)}"
            )
            # Truncate or pad if necessary, though this shouldn't happen with correct logic
            min_len = min(len(test_ids), len(all_predictions))
            test_ids = test_ids[:min_len]
            all_predictions = all_predictions[:min_len]

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": all_predictions})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    runner = Runner()
    runner.train()
    runner.generate_submission()


# Note: The prompt forbids "if __name__ == '__main__':", so we just define the class and main function.
# To execute, one would import this module and call main().
