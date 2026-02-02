import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, get_device
from library.model import PCDRHNet
from library.loss import MaskedL1Loss
from library.dataset import get_dataloaders


class Trainer:
    """
    Handles the training, validation, and inference lifecycle of the PCDRH-Net model.
    Implements the Monotonic Optimization Regime with ReduceLROnPlateau and Early Stopping.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Optimization components
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

        self.criterion = MaskedL1Loss()

        # State tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        # No progress bar for individual epochs to keep logs clean, or minimal
        for x, mask, y in dataloader:
            x = x.to(self.device)
            mask = mask.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Calculate Logic-Gated Masked Loss
            # mask contains u_out (Stream B)
            loss = self.criterion(preds, y, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item()
            count += 1

        return running_loss / count

    def validate(self, dataloader):
        """
        Runs validation inference.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, mask, y in dataloader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)
                loss = self.criterion(preds, y, mask)

                running_loss += loss.item()
                count += 1

        return running_loss / count

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping and Scheduler stepping.
        """
        print(f"Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpointing & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                # print(f"  -> New best model saved! Loss: {val_loss}")
            else:
                self.patience_counter += 1
                # print(f"  -> No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {self.best_val_loss}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Loading best model for inference...")
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError("No best model found. Did training fail?")

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for x, mask in test_loader:
                x = x.to(self.device)
                # mask is not needed for inference output, but returned by loader

                preds = self.model(x)

                # preds shape: (Batch, Seq_Len, 1)
                # Flatten to (Batch * Seq_Len)
                preds_flat = preds.view(-1).cpu().numpy()
                all_preds.append(preds_flat)

        return np.concatenate(all_preds)


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the entire training and submission pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Data Pipeline...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    print("Initializing PCDRH-Net...")
    model = PCDRHNet().to(device)

    # 4. Training
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Inference
    print("Starting Inference...")
    predictions = trainer.predict(test_loader)

    # 6. Submission Generation
    print("Generating Submission File...")

    # Ensure lengths match
    if len(predictions) != len(test_ids):
        raise ValueError(
            f"Shape mismatch: Predictions ({len(predictions)}) vs IDs ({len(test_ids)})"
        )

    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    # Save
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")

    # Print head for verification
    print("Submission Head:")
    print(submission_df.head())


if __name__ == "__main__":
    # This block is not required by the prompt instructions but facilitates local testing if run directly.
    # The prompt asks for the module class/functions implementation.
    run_training_pipeline()
