import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.model import HPIRVN, FocalLoss


class Trainer:
    """
    Manages the training, validation, evaluation, and inference lifecycle
    for the HPI-RVN model.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = HPIRVN().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.criterion = FocalLoss(alpha=Config.ALPHA, gamma=Config.GAMMA)
        self.best_val_loss = float("inf")

        # Ensure working directory exists for model checkpoints
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in loader:
            # Unpack batch and move to device
            geo = batch["geometry"].to(self.device)
            mot = batch["motion"].to(self.device)
            dyn = batch["dynamics"].to(self.device)
            vis = batch["visual"].to(self.device)
            targets = batch["label"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(geo, mot, dyn, vis)

            # Compute loss
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns average loss, true labels, and predicted probabilities.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                geo = batch["geometry"].to(self.device)
                mot = batch["motion"].to(self.device)
                dyn = batch["dynamics"].to(self.device)
                vis = batch["visual"].to(self.device)
                targets = batch["label"].to(self.device).unsqueeze(1)

                logits = self.model(geo, mot, dyn, vis)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * targets.size(0)
                count += targets.size(0)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        if len(all_targets) > 0:
            y_true = np.vstack(all_targets).flatten()
            y_prob = np.vstack(all_probs).flatten()
        else:
            y_true = np.array([])
            y_prob = np.array([])

        return avg_loss, y_true, y_prob

    def optimize_threshold(self, y_true, y_prob):
        """
        Performs a grid search to find the threshold that maximizes MCC.
        """
        best_threshold = 0.5
        best_mcc = -1.0

        thresholds = np.linspace(
            Config.THRESHOLD_SEARCH_START,
            Config.THRESHOLD_SEARCH_END,
            Config.THRESHOLD_SEARCH_STEPS,
        )

        # Filter out 0 and 1 to avoid edge cases if desired, but linspace includes them.
        # We iterate through them.
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        return best_threshold, best_mcc

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    ):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs on device {self.device}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_y, val_probs = self.validate(val_loader)

            # Calculate MCC at default threshold 0.5 for monitoring
            val_pred_default = (val_probs >= 0.5).astype(int)
            val_mcc_default = compute_mcc(val_y, val_pred_default)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC (0.5): {val_mcc_default:.6f}"
            )

            # Early Stopping Logic based on Validation Loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> New best model saved (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model for final threshold optimization
        print("Loading best model for threshold optimization...")
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )

        # Optimize Threshold
        print("Optimizing decision threshold on validation set...")
        _, val_y, val_probs = self.validate(val_loader)
        best_thresh, best_mcc = self.optimize_threshold(val_y, val_probs)

        print(
            f"Optimization Complete. Best Threshold: {best_thresh:.4f}, Best Validation MCC: {best_mcc:.6f}"
        )

        return best_thresh

    def predict(self, loader):
        """
        Generates probability predictions for a dataloader.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                geo = batch["geometry"].to(self.device)
                mot = batch["motion"].to(self.device)
                dyn = batch["dynamics"].to(self.device)
                vis = batch["visual"].to(self.device)

                logits = self.model(geo, mot, dyn, vis)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        if len(all_probs) > 0:
            return np.vstack(all_probs).flatten()
        return np.array([])

    def generate_submission(self, test_loader, test_ids, threshold):
        """
        Generates the submission file using the trained model and optimized threshold.
        """
        print(f"Generating predictions for test set using threshold {threshold:.4f}...")

        probs = self.predict(test_loader)

        # Apply threshold
        predictions = (probs >= threshold).astype(int)

        # Construct dataframe
        submission_df = test_ids.copy()
        submission_df["contact"] = predictions

        # Save
        out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
        print(submission_df.head())
