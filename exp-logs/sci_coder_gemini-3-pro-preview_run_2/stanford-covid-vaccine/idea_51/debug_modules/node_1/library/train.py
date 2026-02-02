import os
import time
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.data import get_dataloaders
from library.model import DSPFN
from library.loss import MCRMSELoss


class Trainer:
    """
    Manages the training lifecycle for the Decoupled-Stem Pure-Feedback Network (DS-PFN).
    Implements the Stabilized Recurrent Loop training strategy.
    """

    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # Initialize Model
        self.model = DSPFN().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # Initialize Loss
        self.criterion = MCRMSELoss().to(self.device)

        # State for Early Stopping
        self.best_val_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training using the Stabilized Recurrent Loop.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs = batch["inputs"].to(self.device)
            partner_map = batch["partner_map"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # --- Stabilized Recurrent Loop ---

            # Step 1: Static Feature Extraction
            # Compute Z once.
            z = self.model.encode(inputs)

            # Step 2: Pass 1 (Zero Initialization)
            # Initialize feedback with zeros
            batch_size, _, length = inputs.shape
            y_0 = torch.zeros((batch_size, 5, length), device=self.device)

            # Decode to get intermediate predictions Y1
            y_1 = self.model.decode(z, y_0, partner_map)

            # Step 3: Pass 2 (Feedback)
            # Detach Y1 to stop gradients flowing back through the feedback loop
            # The model's decode method handles strict masking of unscored targets internally
            r = y_1.detach()

            # Decode using recycled predictions to get Y2
            y_2 = self.model.decode(z, r, partner_map)

            # --- Loss Calculation ---
            # L_total = MCRMSE(Y2) + 0.5 * MCRMSE(Y1)
            loss_main = self.criterion(y_2, targets, mask)
            loss_aux = self.criterion(y_1, targets, mask)
            loss = loss_main + 0.5 * loss_aux

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / max(1, num_batches)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using the 2-pass inference strategy.
        """
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                partner_map = batch["partner_map"].to(self.device)
                targets = batch["targets"].to(self.device)
                # mask is not strictly needed for inference logic but used for metric calculation if needed,
                # though MetricTracker usually takes raw predictions and targets.
                # However, MCRMSE calculation usually focuses on scored columns and positions.
                # The MetricTracker in library.utils assumes inputs are already filtered or handles it?
                # Checking library.utils: MetricTracker updates with y_pred, y_true.
                # It does NOT apply masking internally. We must slice/mask before passing to tracker
                # OR rely on the fact that we only care about scored columns.
                # The prompt description for Metric says: "Positions greater than seq_scored... are not scored".
                # We should slice predictions and targets to Config.SEQ_SCORED and Config.SCORED_INDICES
                # before passing to MetricTracker to be accurate to the competition metric.

                # --- Inference ---
                z = self.model.encode(inputs)

                batch_size, _, length = inputs.shape
                y_0 = torch.zeros((batch_size, 5, length), device=self.device)

                # Pass 1
                y_1 = self.model.decode(z, y_0, partner_map)

                # Pass 2
                y_2 = self.model.decode(z, y_1, partner_map)

                # --- Metric Preparation ---
                # Select Scored Columns
                scored_indices = Config.SCORED_INDICES
                y_pred_scored = y_2[:, scored_indices, :]
                y_true_scored = targets[:, scored_indices, :]

                # Permute to (N, L, C) for slicing
                y_pred_scored = y_pred_scored.permute(0, 2, 1)
                y_true_scored = y_true_scored.permute(0, 2, 1)

                # Slice to Scored Length
                eff_len = Config.SEQ_SCORED
                y_pred_final = y_pred_scored[:, :eff_len, :]
                y_true_final = y_true_scored[:, :eff_len, :]

                tracker.update(y_pred_final, y_true_final)

        return tracker.compute()

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_score)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "  # Full precision
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved (Score: {val_score})")
            else:
                self.patience_counter += 1
                print(f"  >>> Patience: {self.patience_counter}/{Config.PATIENCE}")
                if self.patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val Score: {self.best_val_score}")


def train_model(load_cached_data=True, epochs=Config.EPOCHS):
    """
    Entry point function to train the model.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Data
    print("Loading DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    trainer.fit(train_loader, val_loader, epochs=epochs)
