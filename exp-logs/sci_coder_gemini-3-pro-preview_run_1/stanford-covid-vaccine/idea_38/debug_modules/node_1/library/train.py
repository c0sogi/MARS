import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.loss import MaskedMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


class Trainer:
    """
    Manages the training, validation, and inference process for the RNA degradation model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE

        # Optimization components
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS)
        self.criterion = MaskedMSELoss(seq_scored=config.SEQ_SCORED)

        # Tracking
        self.best_mcrmse = float("inf")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            sequences = batch["sequence"].to(self.device)
            loop_types = batch["loop_type"].to(self.device)
            pair_dists = batch["pair_dist"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(sequences, loop_types, pair_dists)

            # Calculate loss (Masked MSE)
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set and calculates MCRMSE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                sequences = batch["sequence"].to(self.device)
                loop_types = batch["loop_type"].to(self.device)
                pair_dists = batch["pair_dist"].to(self.device)
                targets = batch["target"].to(self.device)

                preds = self.model(sequences, loop_types, pair_dists)

                # Collect data for metric calculation
                # We only score the first seq_scored positions
                preds_scored = preds[:, : self.config.SEQ_SCORED, :]
                targets_scored = targets[:, : self.config.SEQ_SCORED, :]

                all_preds.append(preds_scored.cpu().numpy())
                all_targets.append(targets_scored.cpu().numpy())

        # Concatenate all batches
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE
        score = calculate_mcrmse(y_true, y_pred)
        return score

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_mcrmse = self.validate()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_mcrmse} | "
                f"LR: {current_lr:.6f}"
            )

            # Save best model
            if val_mcrmse < self.best_mcrmse:
                print(
                    f"Validation score improved ({self.best_mcrmse} --> {val_mcrmse}). Saving model..."
                )
                self.best_mcrmse = val_mcrmse
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)

        print(f"Training complete. Best MCRMSE: {self.best_mcrmse}")

    def generate_submission(self):
        """
        Generates predictions for the test set and creates the submission file.
        """
        print("Generating submission...")

        # Load best model weights
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: Best model checkpoint not found. Using current weights.")

        self.model.eval()
        results = []

        # Submission columns required
        # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        with torch.no_grad():
            for batch in self.test_loader:
                sequences = batch["sequence"].to(self.device)
                loop_types = batch["loop_type"].to(self.device)
                pair_dists = batch["pair_dist"].to(self.device)
                ids = batch["id"]

                # Forward pass - predict for full sequence length (107)
                preds = self.model(sequences, loop_types, pair_dists)
                preds = preds.cpu().numpy()  # (Batch, 107, 3)

                # Iterate over samples in batch
                for i, sample_id in enumerate(ids):
                    sample_preds = preds[i]  # (107, 3)

                    # The model predicts: [reactivity, deg_Mg_pH10, deg_Mg_50C]
                    # We need to fill: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
                    # Missing columns: deg_pH10, deg_50C -> fill with 0.0

                    for seqpos in range(self.config.SEQ_LENGTH):
                        row_id = f"{sample_id}_{seqpos}"

                        reactivity = float(sample_preds[seqpos, 0])
                        deg_Mg_pH10 = float(sample_preds[seqpos, 1])
                        deg_Mg_50C = float(sample_preds[seqpos, 2])

                        # Zero-filled columns
                        deg_pH10 = 0.0
                        deg_50C = 0.0

                        results.append(
                            {
                                "id_seqpos": row_id,
                                "reactivity": reactivity,
                                "deg_Mg_pH10": deg_Mg_pH10,
                                "deg_pH10": deg_pH10,
                                "deg_Mg_50C": deg_Mg_50C,
                                "deg_50C": deg_50C,
                            }
                        )

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Ensure column order
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        submission_df = submission_df[cols]

        # Save
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # 3. Model
    print("Initializing Model...")
    model = RNAModel(config=Config)
    model.to(Config.DEVICE)

    # 4. Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, Config)

    # 5. Execute
    trainer.fit()
    trainer.generate_submission()


if __name__ == "__main__":
    main()
