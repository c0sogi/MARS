import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import StabilizedWideResBiGRU
from library.loss import MaskedMSELoss
from library.utils import set_seed, mcrmse_metric


class Trainer:
    """
    Handles the training, validation, and inference lifecycle of the RNA degradation model.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Model
        self.model = StabilizedWideResBiGRU().to(self.device)

        # Optimizer: AdamW with low weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        # Loss Function: Masked MSE
        self.criterion = MaskedMSELoss()

        # Setup Directories
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(self.device)
            loop = batch["loop"].to(self.device)
            dist = batch["dist"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(seq, loop, dist)

            # Compute loss only on masked positions
            loss = self.criterion(preds, targets, mask)

            loss.backward()

            # Critical: Gradient Clipping to stabilize 512-width network
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using MCRMSE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                targets = batch["targets"].to(self.device)

                preds = self.model(seq, loop, dist)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE using the utility function
        score = mcrmse_metric(all_targets, all_preds, pred_len=Config.PRED_LEN)
        return score

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        set_seed(42)
        print(f"Training on device: {self.device}")

        # Load Data
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=load_cached_data
        )

        best_mcrmse = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_mcrmse = self.validate(val_loader)

            self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
            )

            # Checkpoint and Early Stopping
            if val_mcrmse < best_mcrmse:
                best_mcrmse = val_mcrmse
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best MCRMSE: {best_mcrmse}")

        # Generate final submission
        self.generate_submission(test_loader)

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Load best model weights
        if not os.path.exists(self.best_model_path):
            print("No best model found, skipping submission.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(self.device)
                loop = batch["loop"].to(self.device)
                dist = batch["dist"].to(self.device)
                ids = batch["id"]

                preds = self.model(seq, loop, dist)
                preds = preds.cpu().numpy()  # Shape: (B, 107, 3)

                for i, sample_id in enumerate(ids):
                    for pos in range(Config.SEQ_LEN):
                        row_id = f"{sample_id}_{pos}"

                        # Extract predictions for the 3 scored columns
                        # Columns: reactivity, deg_Mg_pH10, deg_Mg_50C
                        p_react = preds[i, pos, 0]
                        p_mg_ph10 = preds[i, pos, 1]
                        p_mg_50c = preds[i, pos, 2]

                        # Unscored columns (deg_pH10, deg_50C) are set to 0.0
                        submission_data.append(
                            {
                                "id_seqpos": row_id,
                                "reactivity": p_react,
                                "deg_Mg_pH10": p_mg_ph10,
                                "deg_pH10": 0.0,
                                "deg_Mg_50C": p_mg_50c,
                                "deg_50C": 0.0,
                            }
                        )

        # Create DataFrame
        sub_df = pd.DataFrame(submission_data)

        # Ensure correct column order
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        sub_df = sub_df[cols]

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(load_cached_data=True, debug=False):
    """
    Entry point function to initialize Trainer and start the process.
    """
    trainer = Trainer(debug=debug)
    trainer.fit(load_cached_data=load_cached_data)
