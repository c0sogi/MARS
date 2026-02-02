import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric, AverageMeter
from library.dataset import prepare_data
from library.model import DFLB_BiLSTM


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the DFLB-BiLSTM model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Stretched Horizon: T_max matches EPOCHS)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.SCHEDULER_T_MAX,
            eta_min=Config.SCHEDULER_ETA_MIN,
        )

        # Loss Function
        self.criterion = WeightedL1Loss()

        # State
        self.best_val_mae = float("inf")

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for x, y, u_out in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            u_out = u_out.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Calculate loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping to stabilize deep RNN training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), x.size(0))

        # Step the scheduler at the end of the epoch
        self.scheduler.step()

        return loss_meter.avg

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        mae_meter = AverageMeter()

        with torch.no_grad():
            for x, y, u_out in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                u_out = u_out.to(self.device)

                # Forward pass
                preds = self.model(x)

                # Calculate Metric (MAE on inspiratory phase)
                mae = compute_metric(preds, y, u_out)

                # Update metrics
                mae_meter.update(mae, x.size(0))

        return mae_meter.avg

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop.
        """
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_mae = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Save best model
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved from {self.best_val_mae} to {val_mae}. Saving model..."
                )
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)

        print(f"Training complete. Best Val MAE: {self.best_val_mae}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x.to(self.device)

                # Forward pass
                preds = self.model(x)

                # Flatten predictions (Batch, Seq_Len) -> (Batch * Seq_Len)
                preds_flat = preds.view(-1).cpu().numpy()
                all_preds.append(preds_flat)

        return np.concatenate(all_preds)


def generate_submission(predictions):
    """
    Maps predictions to IDs and saves the submission file.
    """
    print("Generating submission file...")

    # Load test metadata
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # The dataset was prepared by sorting by breath_id and then id.
    # We must sort the metadata in the same way to align with the flattened predictions.
    test_meta_sorted = test_meta.sort_values(["breath_id", "id"])

    # Assign predictions
    if len(predictions) != len(test_meta_sorted):
        raise ValueError(
            f"Mismatch in prediction count: {len(predictions)} vs {len(test_meta_sorted)}"
        )

    test_meta_sorted["pressure"] = predictions

    # Sort by ID for the final submission format (usually required to be sorted by ID)
    submission = test_meta_sorted.sort_values("id")[["id", "pressure"]]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Prepare Data
    # prepare_data handles caching internally
    train_loader, val_loader, test_loader = prepare_data(debug=Config.DEBUG)

    # Determine input dimension from the dataset
    # Shape of X is (Batch, Seq_Len, Features)
    # We can get features from the first batch
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]
    print(f"Input dimension: {input_dim}")

    # Initialize Model
    device = torch.device(Config.DEVICE)
    model = DFLB_BiLSTM(input_dim=input_dim).to(device)

    # Initialize Trainer
    trainer = Trainer(model, device)

    # Train
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Inference
    predictions = trainer.predict(test_loader)

    # Generate Submission
    generate_submission(predictions)


if __name__ == "__main__":
    main()
