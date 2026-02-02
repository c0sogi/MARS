import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.dataset import SoADataset, SoACollator
from library.model import DMPNN
from library.losses import LossComputer
from library.data_processor import DataProcessor


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Scalar Coupling Prediction task.
    """

    def __init__(self):
        self.config = Config
        self.device = torch.device(self.config.DEVICE)

        # Ensure reproducibility
        self.config.set_seed(self.config.SEED)

        # Create working directories if they don't exist
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

        # Initialize Data Processor and ensure data is ready
        # This checks for cached files internally
        print("Initializing Data Processor...")
        self.processor = DataProcessor()
        self.processor.run(load_cached_data=True)

        # Initialize Datasets and Loaders
        print("Initializing Datasets...")
        self.train_dataset = SoADataset(split="train", load_cached_data=True)
        self.val_dataset = SoADataset(split="val", load_cached_data=True)

        # Collators need reference to their specific dataset to slice arrays
        self.train_collator = SoACollator(self.train_dataset)
        self.val_collator = SoACollator(self.val_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self.train_collator,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=self.val_collator,
            pin_memory=True,
        )

        # Initialize Model
        print("Initializing Model...")
        self.model = DMPNN(
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_rbf=self.config.NUM_RBF,
            num_angle_rbf=self.config.NUM_ANGLE_RBF,
            rbf_gamma=self.config.RBF_GAMMA,
            dropout=self.config.DROPOUT,
        ).to(self.device)

        # Initialize Loss Computer (handles standardization and metrics)
        self.loss_computer = LossComputer().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Cosine Annealing Warm Restarts
        # T_0 is set to a fraction of epochs or a fixed number of steps
        # Here we set it to restart every 10 epochs roughly, or just use CosineAnnealingLR for simplicity in one cycle
        # Given the prompt mentions "refine within 24 hours", WarmRestarts is good for escaping local minima.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Best metric tracking
        self.best_metric = float("inf")
        self.best_model_path = os.path.join(
            self.config.CHECKPOINT_DIR, "best_model.pth"
        )

    def _to_device(self, batch):
        """Moves a dictionary batch to the configured device."""
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(self.device, non_blocking=True)
            else:
                new_batch[k] = v
        return new_batch

    def train(self):
        """
        Executes the training loop with validation and early stopping.
        """
        print(f"Starting training on {self.device}...")

        patience = 5
        patience_counter = 0

        for epoch in range(1, self.config.MAX_EPOCHS + 1):
            start_time = time.time()

            # ==========================
            # Training Phase
            # ==========================
            self.model.train()
            train_losses = []

            for i, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)

                self.optimizer.zero_grad()

                # Mixed Precision Forward Pass
                with autocast():
                    predictions = self.model(batch)
                    loss, components = self.loss_computer(predictions, batch)

                # Backward Pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ==========================
            # Validation Phase
            # ==========================
            self.model.eval()
            val_losses = []
            val_metrics = []

            with torch.no_grad():
                for batch in self.val_loader:
                    batch = self._to_device(batch)

                    with autocast():
                        predictions = self.model(batch)
                        loss, _ = self.loss_computer(predictions, batch)
                        metric = self.loss_computer.compute_metric(predictions, batch)

                    val_losses.append(loss.item())
                    val_metrics.append(metric)

            avg_val_loss = np.mean(val_losses)
            # Metric is LogMAE, averaged across batches (approximate) or computed globally.
            # Since compute_metric returns the mean LogMAE for the batch, averaging them is a reasonable approximation
            # for monitoring.
            avg_val_metric = np.mean(val_metrics)

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.MAX_EPOCHS} | "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Val LogMAE: {avg_val_metric}"
            )

            # ==========================
            # Checkpointing & Early Stopping
            # ==========================
            if avg_val_metric < self.best_metric:
                print(
                    f"Validation metric improved from {self.best_metric} to {avg_val_metric}. Saving model."
                )
                self.best_metric = avg_val_metric
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to submission.csv.
        """
        print("Starting inference on Test set...")

        # Load Test Data
        test_dataset = SoADataset(split="test", load_cached_data=True)
        test_collator = SoACollator(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=test_collator,
            pin_memory=True,
        )

        # Load Best Model
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(
                f"Best model not found at {self.best_model_path}. Train first."
            )

        print(f"Loading best model from {self.best_model_path}...")
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = self._to_device(batch)

                with autocast():
                    predictions = self.model(batch)
                    pred_coupling_std = predictions["coupling"]

                # Unstandardize predictions
                # We need the coupling types from the batch to look up the correct mean/std
                coupling_types = batch["coupling_type"]
                pred_coupling_phys = self.loss_computer.unstandardize(
                    pred_coupling_std, coupling_types
                )

                # Collect results
                all_ids.append(batch["coupling_id"].cpu().numpy())
                all_preds.append(pred_coupling_phys.float().cpu().numpy())

        # Concatenate all batches
        final_ids = np.concatenate(all_ids)
        final_preds = np.concatenate(all_preds)

        # Create Submission DataFrame
        df_sub = pd.DataFrame(
            {"id": final_ids, "scalar_coupling_constant": final_preds}
        )

        # Sort by ID to match sample submission format (optional but good practice)
        df_sub = df_sub.sort_values("id")

        # Save
        save_path = self.config.SUBMISSION_PATH
        print(f"Saving submission to {save_path}...")
        df_sub.to_csv(save_path, index=False)
        print("Submission generation complete.")
