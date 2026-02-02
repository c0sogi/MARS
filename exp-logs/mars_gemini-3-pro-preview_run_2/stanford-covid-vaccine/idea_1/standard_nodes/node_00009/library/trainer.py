import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.model import RNAResNet
from library.dataset import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the RNA degradation model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config=Config,
        device=None,
    ):
        self.config = config
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Optimization
        self.criterion = MCRMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Loss calculation (MCRMSELoss handles slicing internally)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss

    def fit(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss} | "  # Printing full precision as requested
                f"LR: {current_lr:.2e}"
            )

            # Early Stopping and Model Saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                # print(f"  -> Saved best model (Loss: {val_loss})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.

        Returns:
            np.ndarray: Predictions of shape (Num_Samples, Seq_Len, Num_Targets)
        """
        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            self.model.eval()
        else:
            print("Warning: No best model found. Using current weights.")

        all_preds = []

        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu().numpy())

        # Concatenate all batches: (N, 107, 5)
        return np.concatenate(all_preds, axis=0)

    def generate_submission(self, predictions):
        """
        Formats predictions into the competition submission CSV.

        Args:
            predictions (np.ndarray): Array of shape (Num_Samples, 107, 5)
        """
        print("Generating submission file...")

        # Get IDs from the test dataset
        # Note: process_data returns ids, which are stored in the dataset
        test_ids = self.test_loader.dataset.ids

        submission_data = []
        target_cols = self.config.TARGET_COLS

        # Iterate through each sample
        for i, sample_id in enumerate(test_ids):
            # Get predictions for this sample: shape (107, 5)
            sample_preds = predictions[i]

            # Create rows for each sequence position (0 to 106)
            for seqpos in range(self.config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()

                # Construct row: [id_seqpos, val1, val2, val3, val4, val5]
                submission_data.append([row_id] + row_values)

        # Create DataFrame
        cols = ["id_seqpos"] + target_cols
        submission_df = pd.DataFrame(submission_data, columns=cols)

        # Save
        save_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")


def run_training(
    max_samples=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Main entry point to run the training pipeline.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.create_dirs()

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAResNet(
        input_channels=Config.INPUT_CHANNELS, num_targets=Config.NUM_TARGETS
    )

    # 4. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=Config,
    )

    # 5. Training
    trainer.fit(epochs=epochs)

    # 6. Prediction
    print("Generating predictions...")
    predictions = trainer.predict()

    # 7. Submission
    trainer.generate_submission(predictions)
