import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    SEED,
    METADATA_DIR,
)
from library.utils import seed_everything, get_device
from library.loss import FocalLoss

# Ensure reproducibility
seed_everything(SEED)


class Trainer:
    """
    Manages training, evaluation, threshold optimization, and inference
    for the Entity-Augmented Residual-Visual Network.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device else get_device()

        self.model.to(self.device)
        self.best_threshold = 0.5
        self.best_mcc = -1.0

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            x_kin = data["kinematic"].to(self.device)
            x_cat = data["categorical"].to(self.device)
            x_vis = data["visual"].to(self.device)
            x_gate = data["gating"].to(self.device)
            y = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x_kin, x_cat, x_vis, x_gate)

            # Compute loss
            loss = self.criterion(logits, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        return running_loss / count if count > 0 else 0.0

    def evaluate(self, loader):
        """
        Evaluates the model on a given loader.
        Returns average loss, raw logits (numpy), and true labels (numpy).
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                x_kin = data["kinematic"].to(self.device)
                x_cat = data["categorical"].to(self.device)
                x_vis = data["visual"].to(self.device)
                x_gate = data["gating"].to(self.device)
                y = target.to(self.device)

                logits = self.model(x_kin, x_cat, x_vis, x_gate)
                loss = self.criterion(logits, y)

                running_loss += loss.item() * y.size(0)
                count += y.size(0)

                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        if len(all_logits) > 0:
            all_logits = np.concatenate(all_logits)
            all_targets = np.concatenate(all_targets)
        else:
            all_logits = np.array([])
            all_targets = np.array([])

        return avg_loss, all_logits, all_targets

    def optimize_threshold(self, logits, targets):
        """
        Performs a grid search to find the threshold that maximizes MCC.
        """
        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))

        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        # Vectorized calculation could be memory intensive for large N,
        # so we loop through thresholds efficiently.
        # Given N ~800k, looping 100 times is fast enough (seconds).

        # Flatten targets
        targets = targets.flatten()
        probs = probs.flatten()

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = matthews_corrcoef(targets, preds)

            if score > best_mcc:
                best_mcc = score
                best_thresh = t

        return best_thresh, best_mcc

    def run(self, epochs=EPOCHS, patience=PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_val_mcc = -1.0
        patience_counter = 0
        best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_logits, val_targets = self.evaluate(self.val_loader)

            # Optimize Threshold on Validation Set
            curr_threshold, curr_mcc = self.optimize_threshold(val_logits, val_targets)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {curr_mcc:.10f} (Threshold: {curr_threshold:.2f})"
            )

            # Early Stopping Check
            if curr_mcc > best_val_mcc:
                best_val_mcc = curr_mcc
                self.best_threshold = curr_threshold
                self.best_mcc = curr_mcc
                patience_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                # Save best threshold
                np.save(
                    os.path.join(WORKING_DIR, "best_threshold.npy"),
                    np.array([curr_threshold]),
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(
            f"Training complete. Best Val MCC: {self.best_mcc:.10f} at Threshold: {self.best_threshold:.2f}"
        )

    def predict_and_submit(self):
        """
        Loads the best model, runs inference on test set, applies optimal threshold,
        and generates submission file.
        """
        print("Generating submission...")

        # Load best model
        best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model weights.")
        else:
            print("Warning: Best model not found, using current weights.")

        # Load best threshold
        thresh_path = os.path.join(WORKING_DIR, "best_threshold.npy")
        if os.path.exists(thresh_path):
            self.best_threshold = float(np.load(thresh_path)[0])
            print(f"Loaded optimized threshold: {self.best_threshold:.4f}")

        # Inference
        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for data in self.test_loader:
                # Test loader might return (data, placeholder_y) or just data depending on implementation
                # Dataset __getitem__ returns (data, y) if y is not None.
                # In get_dataloaders, test_dataset has y (zeros).
                if isinstance(data, (list, tuple)):
                    data = data[0]  # Extract input dict

                x_kin = data["kinematic"].to(self.device)
                x_cat = data["categorical"].to(self.device)
                x_vis = data["visual"].to(self.device)
                x_gate = data["gating"].to(self.device)

                logits = self.model(x_kin, x_cat, x_vis, x_gate)
                all_logits.append(logits.cpu().numpy())

        if len(all_logits) > 0:
            all_logits = np.concatenate(all_logits).flatten()
        else:
            all_logits = np.array([])

        # Apply Threshold
        probs = 1.0 / (1.0 + np.exp(-all_logits))
        predictions = (probs >= self.best_threshold).astype(int)

        # Load Test Metadata to get Contact IDs
        # We use metadata/test.csv which is derived from sample_submission.csv
        # and aligned with the test_loader order.
        test_meta_path = os.path.join(METADATA_DIR, "test.csv")
        df_test = pd.read_csv(test_meta_path)

        if len(df_test) != len(predictions):
            print(
                f"Error: Mismatch between metadata rows ({len(df_test)}) and predictions ({len(predictions)})"
            )
            # Fallback: trim or pad? This should not happen if loaders are correct.
            # We will raise error to be safe.
            raise ValueError("Prediction count mismatch.")

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")


def train_model(model, train_loader, val_loader, test_loader):
    """
    Helper function to instantiate Trainer and run the pipeline.
    """
    device = get_device()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # Loss
    criterion = FocalLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Run Training
    trainer.run()

    # Generate Submission
    trainer.predict_and_submit()
