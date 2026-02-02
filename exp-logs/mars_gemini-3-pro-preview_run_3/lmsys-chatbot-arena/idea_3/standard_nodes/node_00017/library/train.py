import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data import get_dataloaders
from library.model import SiameseBiLSTMAttention


class Trainer:
    """
    Trainer class to manage the training, validation, and inference lifecycle
    of the SiameseBiLSTMAttention.
    """

    def __init__(self):
        # 1. Setup
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        print(f"Initializing Trainer on device: {self.device}")

        # 2. Data
        print("Loading dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=True
        )

        # 3. Model
        print("Initializing model...")
        self.model = SiameseBiLSTMAttention()
        self.model.to(self.device)

        # 4. Optimization
        # Calculate total training steps for scheduler
        num_update_steps_per_epoch = (
            len(self.train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
        )
        max_train_steps = Config.EPOCHS * num_update_steps_per_epoch

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-3,  # Higher LR for LSTM
            weight_decay=1e-5,
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(
                max_train_steps * 0.0
            ),  # No warmup needed for LSTM usually
            num_training_steps=max_train_steps,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=Config.USE_FP16)

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        optimizer_step_count = 0

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            ids_prompt = batch["ids_prompt"].to(self.device)
            ids_a = batch["ids_a"].to(self.device)
            ids_b = batch["ids_b"].to(self.device)
            scalar_features = batch["scalar_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                logits = self.model(
                    ids_prompt=ids_prompt,
                    ids_a=ids_a,
                    ids_b=ids_b,
                    scalar_features=scalar_features,
                )

                # Check if labels are one-hot/probabilities or indices
                # CrossEntropyLoss expects class indices if target is Long,
                # or probabilities if target is Float.
                # Our labels are probabilities (float32).
                loss = self.criterion(logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            # Backward Pass
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

            # Optimizer Step (Gradient Accumulation)
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

                # Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                optimizer_step_count += 1

        avg_loss = total_loss / len(self.train_loader)
        duration = time.time() - start_time
        print(
            f"Epoch {epoch_idx+1} | Train Loss: {avg_loss:.6f} | Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                ids_prompt = batch["ids_prompt"].to(self.device)
                ids_a = batch["ids_a"].to(self.device)
                ids_b = batch["ids_b"].to(self.device)
                scalar_features = batch["scalar_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass (no autocast needed for eval usually, but good for consistency/speed)
                with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                    logits = self.model(
                        ids_prompt=ids_prompt,
                        ids_a=ids_a,
                        ids_b=ids_b,
                        scalar_features=scalar_features,
                    )

                # Apply Softmax to get probabilities
                probs = torch.softmax(logits, dim=1)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute Log Loss
        val_loss = compute_log_loss(all_targets, all_preds)
        return val_loss

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_val_loss = float("inf")
        patience = 2  # Number of epochs to wait for improvement
        patience_counter = 0

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch+1} | Validation Log Loss: {val_loss}")

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
                )
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Log Loss: {best_val_loss}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model
        and creates the submission file.
        """
        print("Generating submission...")

        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                ids_prompt = batch["ids_prompt"].to(self.device)
                ids_a = batch["ids_a"].to(self.device)
                ids_b = batch["ids_b"].to(self.device)
                scalar_features = batch["scalar_features"].to(self.device)
                ids = batch["id"]

                with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                    logits = self.model(
                        ids_prompt=ids_prompt,
                        ids_a=ids_a,
                        ids_b=ids_b,
                        scalar_features=scalar_features,
                    )

                # Cast to float32 for numerical stability before softmax
                probs = torch.softmax(logits.float(), dim=1)

                all_ids.extend(ids.numpy())
                all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {
                "id": all_ids,
                "winner_model_a": all_preds[:, 0],
                "winner_model_b": all_preds[:, 1],
                "winner_tie": all_preds[:, 2],
            }
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False, float_format="%.15f")
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_pipeline():
    """
    Entry point to run the full training and submission pipeline.
    """
    trainer = Trainer()
    trainer.fit()
    trainer.generate_submission()
