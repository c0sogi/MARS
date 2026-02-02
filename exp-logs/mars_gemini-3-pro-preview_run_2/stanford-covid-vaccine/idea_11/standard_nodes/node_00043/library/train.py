import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, MCRMSELoss, GlobalMetricTracker
from library.data import get_dataloaders
from library.model import CascadedDenseNet


class Trainer:
    """
    Handles the training, validation, and checkpointing of the CascadedDenseNet.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function with masking for scored columns
        self.criterion = MCRMSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.FACTOR,
            patience=Config.PATIENCE,
        )

        self.best_score = float("inf")

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            inputs = batch["inputs"].to(self.device)
            partners = batch["partner_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, partners)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """Runs validation and returns the global MCRMSE score."""
        self.model.eval()
        tracker = GlobalMetricTracker()

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                partners = batch["partner_indices"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(inputs, partners)

                # Update global metric tracker
                tracker.update(outputs, targets)

        return tracker.get_score()

    def fit(self, epochs, es_patience):
        """
        Main training loop with Early Stopping.
        """
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_score = self.validate()

            end_time = time.time()
            duration = end_time - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val MCRMSE: {val_score:.10f} | "
                f"Time: {duration:.2f}s"
            )

            # Scheduler step
            self.scheduler.step(val_score)

            # Checkpointing and Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= es_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break


def train_model(debug=False):
    """
    Initializes the environment, data, and model, then runs the training process.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached_data=True)

    # Device
    device = torch.device(Config.DEVICE)

    # Model
    model = CascadedDenseNet().to(device)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run Training
    trainer.fit(Config.EPOCHS, Config.ES_PATIENCE)

    return trainer.best_score


def generate_submission(debug=False):
    """
    Loads the best model, generates predictions for the test set,
    and creates the submission CSV file.
    """
    print("Generating submission...")
    set_seed(Config.SEED)

    # Load Test Data
    _, _, test_loader = get_dataloaders(debug=debug, load_cached_data=True)

    # Load Test Metadata (for IDs)
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Load Model
    device = torch.device(Config.DEVICE)
    model = CascadedDenseNet().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Run training first."
        )

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partners = batch["partner_indices"].to(device)

            # Forward pass (Batch, Seq_Len, Num_Targets)
            outputs = model(inputs, partners)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()
            all_preds.append(preds)

    # Concatenate all batches: (Total_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission Data
    # We need to flatten the predictions to match the id_seqpos format
    # Shape transformation: (N, 107, 5) -> (N*107, 5)

    submission_ids = []

    # Generate ID_seqpos keys
    for sample_id in test_df["id"]:
        for i in range(Config.SEQ_LENGTH):
            submission_ids.append(f"{sample_id}_{i}")

    # Flatten predictions
    # Reshape to (N*107, 5)
    flat_preds = all_preds.reshape(-1, Config.NUM_TARGETS)

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", submission_ids)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
