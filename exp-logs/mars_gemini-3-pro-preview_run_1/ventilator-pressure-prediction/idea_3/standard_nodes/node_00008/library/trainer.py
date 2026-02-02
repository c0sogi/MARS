import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss
from library.dataset import get_data_loaders
from library.model import TransLSTM


class Trainer:
    """
    Trainer class for the Hybrid Transformer-LSTM model.
    Handles training, validation, checkpointing, and inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = TransLSTM().to(self.device)

        # Initialize Loss
        self.criterion = MaskedL1Loss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler will be initialized in fit() after data loading
        self.scheduler = None

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # State tracking
        self.best_loss = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, u_out, y in train_loader:
            x = x.to(self.device)
            u_out = u_out.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast(enabled=(self.device.type == "cuda")):
                preds = self.model(x)
                loss = self.criterion(preds, y, u_out)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()

            # Unscale and Clip Gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, u_out, y in val_loader:
                x = x.to(self.device)
                u_out = u_out.to(self.device)
                y = y.to(self.device)

                with autocast(enabled=(self.device.type == "cuda")):
                    preds = self.model(x)
                    loss = self.criterion(preds, y, u_out)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        seed_everything(Config.SEED)

        # 1. Load Data
        print("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(
            load_cached_data=load_cached_data
        )

        # 2. Initialize Scheduler (OneCycleLR requires steps_per_epoch)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR,
            steps_per_epoch=len(train_loader),
            epochs=Config.EPOCHS,
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train and Validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpointing & Early Stopping
            if val_loss < self.best_loss:
                print(
                    f"Validation loss improved from {self.best_loss} to {val_loss}. Saving model..."
                )
                self.best_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.OUTPUT_DIR, "model.pth"),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        # 3. Generate Predictions after training
        self.predict(test_loader)

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best model.
        Saves submission files.
        """
        print("Loading best model for inference...")
        model_path = os.path.join(Config.OUTPUT_DIR, "model.pth")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for x, u_out, ids in test_loader:
                x = x.to(self.device)

                # Forward pass
                with autocast(enabled=(self.device.type == "cuda")):
                    preds = self.model(x)

                # Collect results
                # preds: [Batch, Seq_Len], ids: [Batch, Seq_Len]
                all_preds.append(preds.float().cpu().numpy().flatten())
                all_ids.append(ids.numpy().flatten())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_ids = np.concatenate(all_ids)

        # Create DataFrame
        submission = pd.DataFrame({"id": all_ids, "pressure": all_preds})

        # Save to Working Directory
        working_sub_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
        submission.to_csv(working_sub_path, index=False)
        print(f"Submission saved to {working_sub_path}")

        # Save to Final Submission Directory
        os.makedirs("./submission", exist_ok=True)
        final_sub_path = "./submission/submission.csv"
        submission.to_csv(final_sub_path, index=False)
        print(f"Final submission saved to {final_sub_path}")
