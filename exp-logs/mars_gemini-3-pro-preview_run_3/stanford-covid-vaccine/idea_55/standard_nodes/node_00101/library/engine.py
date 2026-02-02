import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import compute_mcrmse


class Engine:
    """
    Engine class to handle training, evaluation, and inference for the RNA Degradation Prediction model.
    """

    def __init__(self, model, optimizer, scheduler=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else Config.DEVICE

        # Ensure model is on the correct device
        self.model.to(self.device)

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            features = batch["features"].to(self.device)
            pair_indices = batch["pair_indices"].to(self.device)
            pair_masks = batch["pair_masks"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            outputs = self.model(features, pair_indices, pair_masks)

            # Slice to scored sequence length (first 68 bases) for loss calculation
            # We train on all 5 targets (Multi-Task Learning) as per strategy
            outputs_sliced = outputs[:, : Config.SEQ_SCORED, :]
            targets_sliced = targets[:, : Config.SEQ_SCORED, :]

            # Compute MCRMSE Loss manually to maintain gradients
            # 1. Mean Squared Error per column (averaging over batch and sequence)
            mse = torch.mean((outputs_sliced - targets_sliced) ** 2, dim=(0, 1))

            # 2. Root Mean Squared Error per column (add epsilon for stability)
            rmse = torch.sqrt(mse + 1e-6)

            # 3. Mean of RMSEs (MCRMSE)
            loss = torch.mean(rmse)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Mandatory for hybrid architecture)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # Step scheduler (Cosine Annealing is typically updated per epoch)
        if self.scheduler:
            self.scheduler.step()

        return running_loss / num_batches

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set using the competition metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                pair_masks = batch["pair_masks"].to(self.device)
                targets = batch["targets"]  # Keep targets on CPU for accumulation

                outputs = self.model(features, pair_indices, pair_masks)

                all_preds.append(outputs.cpu())
                all_targets.append(targets)

        if not all_preds:
            return 0.0

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute MCRMSE using the utility function
        # This handles slicing to SEQ_SCORED and selecting SCORED_INDICES
        score = compute_mcrmse(all_preds, all_targets)

        return score

    def predict(self, dataloader):
        """
        Generates predictions for the test set.
        Returns full sequence predictions (107 length) and IDs.
        """
        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                pair_masks = batch["pair_masks"].to(self.device)
                ids = batch["id"]

                outputs = self.model(features, pair_indices, pair_masks)

                # Store full sequence predictions (107 length)
                all_preds.append(outputs.cpu().numpy())
                all_ids.extend(ids)

        if not all_preds:
            return np.array([]), []

        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds, all_ids

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    ):
        """
        Runs the full training process with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.evaluate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
            )

            # Save best model
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        print(f"Training finished. Best Validation Score: {best_score}")

        # Load best model weights for future use/inference
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
