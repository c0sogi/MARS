import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, mcrmse_metric, format_submission
from library.dataset import RNADataset
from library.model import RNATransformer, masked_mse_loss


class Engine:
    """
    Engine class to handle training, evaluation, and inference for the RNA-Transformer model.
    """

    def __init__(self, device=None):
        """
        Initializes the Engine with model, optimizer, and device.

        Args:
            device (torch.device, optional): Device to run the model on. Defaults to Config.DEVICE.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = RNATransformer().to(self.device)

        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler is initialized in run_training once dataset size is known
        self.scheduler = None

    def train_step(self, loader):
        """
        Performs one epoch of training.

        Args:
            loader (DataLoader): Training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # Move batch data to device
            seq = batch["sequence"].to(self.device)
            struct = batch["structure"].to(self.device)
            loop = batch["loop"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(seq, struct, loop)

            # Calculate masked loss
            loss = masked_mse_loss(preds, targets, mask)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def eval_step(self, loader):
        """
        Performs evaluation on the validation set.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            tuple: (Average Validation Loss, Validation MCRMSE Score)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["loop"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                preds = self.model(seq, struct, loop)
                loss = masked_mse_loss(preds, targets, mask)
                total_loss += loss.item()

                # Store predictions and targets for metric calculation
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(loader)

        # Calculate MCRMSE Metric
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        mcrmse = mcrmse_metric(all_targets_tensor, all_preds_tensor)

        return avg_loss, mcrmse

    def run_training(
        self, epochs=Config.EPOCHS, save_path=Config.MODEL_SAVE_PATH, debug=False
    ):
        """
        Runs the full training loop with early stopping.

        Args:
            epochs (int): Number of epochs to train.
            save_path (str): Path to save the best model checkpoint.
            debug (bool): If True, uses a small subset of data for debugging purposes.
        """
        seed_everything(Config.SEED)
        print(f"Starting training on {self.device}...")

        # Load Datasets
        train_dataset = RNADataset(split="train", load_cached_data=True)
        val_dataset = RNADataset(split="val", load_cached_data=True)

        if debug:
            print("Debug mode enabled: utilizing a subset of data (64 samples).")
            train_dataset = Subset(train_dataset, range(min(len(train_dataset), 64)))
            val_dataset = Subset(val_dataset, range(min(len(val_dataset), 64)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Setup Scheduler
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        best_mcrmse = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_step(train_loader)
            val_loss, val_mcrmse = self.eval_step(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCRMSE: {val_mcrmse}"
            )

            # Checkpointing and Early Stopping
            if val_mcrmse < best_mcrmse:
                best_mcrmse = val_mcrmse
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    def inference(
        self, model_path=Config.MODEL_SAVE_PATH, submission_path=Config.SUBMISSION_FILE
    ):
        """
        Runs inference on the test set using the best saved model and generates the submission file.

        Args:
            model_path (str): Path to the saved model checkpoint.
            submission_path (str): Path to save the generated submission CSV.
        """
        seed_everything(Config.SEED)
        print("Starting inference...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load Model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load Test Data
        test_dataset = RNADataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_preds = []
        all_ids = []

        # Generate Predictions
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["loop"].to(self.device)

                preds = self.model(seq, struct, loop)
                all_preds.append(preds.cpu().numpy())
                all_ids.extend(batch["id"])

        # Concatenate predictions
        final_preds = np.concatenate(all_preds, axis=0)

        # Save Submission
        print(f"Saving submission to {submission_path}...")
        format_submission(all_ids, final_preds, submission_path)
        print("Inference done.")
