import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, AverageMeter, kl_divergence_score
from library.data import get_dataloaders
from library.model import SymmetryAwareNet


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the Symmetry-Aware Network.
    """

    def __init__(self, model, device, learning_rate=Config.LEARNING_RATE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
        )
        # KLDivLoss expects log-probabilities as input
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.best_score = float("inf")

    def train_epoch(self, loader, scheduler=None):
        self.model.train()
        loss_meter = AverageMeter()

        for batch in loader:
            # Move data to device
            left_eeg = batch["left_eeg"].to(self.device)
            right_eeg = batch["right_eeg"].to(self.device)
            spec = batch["spectrogram"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(left_eeg, right_eeg, spec)

            # Loss calculation: KLDiv requires log_softmax inputs
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.criterion(log_probs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if scheduler:
                scheduler.step()

            loss_meter.update(loss.item(), left_eeg.size(0))

        return loss_meter.avg

    def validate(self, loader):
        self.model.eval()
        loss_meter = AverageMeter()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in loader:
                left_eeg = batch["left_eeg"].to(self.device)
                right_eeg = batch["right_eeg"].to(self.device)
                spec = batch["spectrogram"].to(self.device)
                targets = batch["label"].to(self.device)

                logits = self.model(left_eeg, right_eeg, spec)

                # Loss
                log_probs = F.log_softmax(logits, dim=1)
                loss = self.criterion(log_probs, targets)
                loss_meter.update(loss.item(), left_eeg.size(0))

                # Predictions for metric calculation (Softmax)
                probs = F.softmax(logits, dim=1)

                preds_list.append(probs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        all_preds = np.concatenate(preds_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Calculate metric using the provided utility
        score = kl_divergence_score(all_targets, all_preds)

        return loss_meter.avg, score

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    ):
        # Setup OneCycleLR Scheduler
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.MAX_LR,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=100.0,
        )

        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, scheduler)
            val_loss, val_score = self.validate(val_loader)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val KL Score: {val_score}"
            )

            # Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with KL Score: {val_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation KL Score: {self.best_score}")

    def predict(self, loader):
        """
        Runs inference on a loader and returns probabilities and eeg_ids.
        """
        self.model.eval()
        preds_list = []
        ids_list = []

        with torch.no_grad():
            for batch in loader:
                left_eeg = batch["left_eeg"].to(self.device)
                right_eeg = batch["right_eeg"].to(self.device)
                spec = batch["spectrogram"].to(self.device)

                # Get eeg_ids for submission mapping
                eeg_ids = (
                    batch["eeg_id"].numpy()
                    if isinstance(batch["eeg_id"], torch.Tensor)
                    else batch["eeg_id"]
                )

                logits = self.model(left_eeg, right_eeg, spec)
                probs = F.softmax(logits, dim=1)

                preds_list.append(probs.cpu().numpy())
                ids_list.append(eeg_ids)

        return np.concatenate(preds_list, axis=0), np.concatenate(ids_list, axis=0)


def train_model(
    debug_size=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.init_directories()

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, debug_size=debug_size
    )

    # 3. Model Initialization
    model = SymmetryAwareNet()

    # 4. Training
    trainer = Trainer(model, Config.DEVICE)
    trainer.fit(train_loader, val_loader, epochs=epochs)

    # 5. Inference & Submission
    print("Generating submission...")

    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # Predict on Test Set
    probs, eeg_ids = trainer.predict(test_loader)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(probs, columns=Config.TARGET_COLS)
    # Rename columns to match submission format (remove _prob suffix if necessary,
    # but config TARGET_COLS are usually prob columns. The task description asks for *_vote columns.
    # However, sample_submission usually expects probabilities in columns named *_vote.
    # Let's map config cols to required submission cols.

    submission_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    submission_df.columns = submission_cols

    # Insert eeg_id at the beginning
    submission_df.insert(0, "eeg_id", eeg_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    return trainer.best_score


if __name__ == "__main__":
    # This block is not required by the prompt instructions but facilitates local testing if run directly.
    # The prompt asks to implement the module functions.
    pass
