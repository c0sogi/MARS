import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloader
from library.model_components import HSDARNModel
from library.loss_metric import MCRMSELoss, GlobalMCRMSE


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the HS-DARN model.
    """

    def __init__(self):
        # Set reproducibility
        Config.set_seed(Config.SEED)
        self.device = Config.DEVICE

        # Initialize Model
        print(f"Initializing HS-DARN Model on {self.device}...")
        self.model = HSDARNModel().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Loss Function
        self.criterion = MCRMSELoss().to(self.device)

        # Checkpoint paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        self.best_score = float("inf")

    def train_one_epoch(self, loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            partner_map = batch["partner_map"].to(self.device)
            targets = batch["targets"].to(self.device)
            batch_size = inputs.size(0)

            self.optimizer.zero_grad()

            # Forward Pass: Returns (y_hat_1, y_hat_2)
            y1, y2 = self.model(inputs, partner_map)

            # Compute Loss: Weighted sum of refined and initial pass
            loss1 = self.criterion(y1, targets)
            loss2 = self.criterion(y2, targets)
            loss = loss2 + 0.5 * loss1

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, loader):
        """
        Runs validation using Global MCRMSE.
        """
        self.model.eval()
        metric = GlobalMCRMSE()

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                partner_map = batch["partner_map"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward Pass: We only care about the final refined prediction y2 for validation
                _, y2 = self.model(inputs, partner_map)

                # Update global metric accumulator
                metric.update(y2, targets)

        return metric.compute()

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")

        train_loader = get_dataloader(
            "train", batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = get_dataloader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

        patience = 5
        counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            # Print metrics with full precision
            print(f"Epoch {epoch}: Train Loss {train_loss}, Val MCRMSE {val_score}")

            # Scheduler Step
            self.scheduler.step(val_score)

            # Early Stopping & Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with MCRMSE: {self.best_score}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_score}")

    def predict(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Starting inference...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            print(f"Loading model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: No best model found. Using current model state.")

        self.model.eval()
        test_loader = get_dataloader(
            "test", batch_size=Config.BATCH_SIZE, shuffle=False
        )

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(self.device)
                partner_map = batch["partner_map"].to(self.device)
                ids = batch["id"]

                # Forward Pass: Use refined prediction y2
                _, y2 = self.model(inputs, partner_map)

                # Move to CPU and store
                all_preds.append(y2.cpu().numpy())
                all_ids.extend(ids)

        # Concatenate all predictions: (N_samples, 107, 5)
        all_preds = np.concatenate(all_preds, axis=0)

        # Format submission
        print("Formatting submission...")
        submission_rows = []
        target_cols = (
            Config.TARGET_COLS
        )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        for i, sample_id in enumerate(all_ids):
            sample_preds = all_preds[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                # Row ID format: id_sequence_position
                row_id = f"{sample_id}_{seqpos}"

                row_data = {"id_seqpos": row_id}

                # Map model output channels to column names
                for j, col in enumerate(target_cols):
                    row_data[col] = float(sample_preds[seqpos, j])

                submission_rows.append(row_data)

        # Create DataFrame
        df_sub = pd.DataFrame(submission_rows)

        # Ensure correct column order
        cols_order = ["id_seqpos"] + target_cols
        df_sub = df_sub[cols_order]

        # Save to CSV
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Entry point to run the full training and inference pipeline.
    """
    trainer = Trainer()
    trainer.fit()
    trainer.predict()
