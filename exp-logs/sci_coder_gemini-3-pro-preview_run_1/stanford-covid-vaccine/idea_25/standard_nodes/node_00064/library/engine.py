import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import DynamicDepthWideStreamBiGRU
from library.loss import MaskedMSELoss
from library.metrics import compute_mcrmse


class Engine:
    """
    Engine class to handle training, validation, and inference for the RNA degradation model.
    Encapsulates the model, optimizer, scheduler, loss function, and training loops.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = DynamicDepthWideStreamBiGRU().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX
        )
        self.criterion = MaskedMSELoss()
        self.best_score = float("inf")

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (seq, loop, dist, targets, mask) in enumerate(train_loader):
            seq = seq.to(self.device)
            loop = loop.to(self.device)
            dist = dist.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(seq, loop, dist, mask)

            # Compute loss
            loss = self.criterion(preds, targets, mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns the MCRMSE score.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for seq, loop, dist, targets, mask in val_loader:
                seq = seq.to(self.device)
                loop = loop.to(self.device)
                dist = dist.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                preds = self.model(seq, loop, dist, mask)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(mask.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute MCRMSE
        score = compute_mcrmse(all_preds, all_targets, all_masks)
        return score

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, early_stopping_patience=5
    ):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.8f} | Val MCRMSE: {val_score:.16f}"
            )

            # Checkpointing and Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"New best model saved with MCRMSE: {self.best_score:.16f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Generating submission...")

        # Load best model weights
        if os.path.exists(Config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded weights from {Config.CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        results = []

        # Target columns predicted by the model (indices in output tensor)
        # 0: reactivity, 1: deg_Mg_pH10, 2: deg_Mg_50C

        with torch.no_grad():
            for seq, loop, dist, mask, sample_ids in test_loader:
                seq = seq.to(self.device)
                loop = loop.to(self.device)
                dist = dist.to(self.device)

                # Model output: (Batch, Seq_Len, 3)
                preds = self.model(seq, loop, dist)
                preds = preds.cpu().numpy()

                # Iterate over batch
                for i, sample_id in enumerate(sample_ids):
                    # For each position in the sequence
                    for pos in range(Config.SEQ_LEN):
                        row_id = f"{sample_id}_{pos}"

                        # Extract predictions
                        val_reactivity = preds[i, pos, 0]
                        val_deg_Mg_pH10 = preds[i, pos, 1]
                        val_deg_Mg_50C = preds[i, pos, 2]

                        # Fill unscored columns with 0.0 as per task requirements
                        val_deg_pH10 = 0.0
                        val_deg_50C = 0.0

                        results.append(
                            {
                                "id_seqpos": row_id,
                                "reactivity": val_reactivity,
                                "deg_Mg_pH10": val_deg_Mg_pH10,
                                "deg_pH10": val_deg_pH10,
                                "deg_Mg_50C": val_deg_Mg_50C,
                                "deg_50C": val_deg_50C,
                            }
                        )

        # Create DataFrame
        df_sub = pd.DataFrame(results)

        # Ensure column order matches sample submission
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        df_sub = df_sub[cols]

        # Save
        df_sub.to_csv(Config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.FINAL_SUBMISSION_PATH}")


def run_pipeline():
    """
    Orchestrates the data loading, training, and prediction process.
    """
    # 1. Setup
    Config.setup()

    # 2. Data Loading (with caching)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Engine
    engine = Engine()

    # 4. Train
    # Using 5 as patience for early stopping
    engine.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, early_stopping_patience=5
    )

    # 5. Predict
    engine.predict(test_loader)
