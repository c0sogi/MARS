import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import random

from library.config import Config
from library.dataset import NFLContactDataset
from library.model import EGRVNet
from library.loss import FocalLoss


class Trainer:
    def __init__(self, device=None):
        """
        Trainer class for the EGRVNet model.
        Encapsulates training, validation, threshold optimization, and submission.
        """
        self.device = device if device else torch.device(Config.DEVICE)

        # Initialize Model
        self.model = EGRVNet().to(self.device)

        # Initialize Loss and Optimizer
        self.criterion = FocalLoss(alpha=Config.ALPHA, gamma=Config.GAMMA)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Paths
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.working_dir, "best_model.pth")
        self.best_thresh_path = os.path.join(self.working_dir, "best_threshold.npy")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            kin = batch["kinematic"].to(self.device)
            vis = batch["visual"].to(self.device)
            cat = batch["categorical"].to(self.device)
            target = batch["target"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(kin, vis, cat)
            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Runs validation and returns loss, predictions, and targets.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                kin = batch["kinematic"].to(self.device)
                vis = batch["visual"].to(self.device)
                cat = batch["categorical"].to(self.device)
                target = batch["target"].to(self.device).unsqueeze(1)

                logits = self.model(kin, vis, cat)
                loss = self.criterion(logits, target)
                running_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_loss = running_loss / len(val_loader)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        return avg_loss, all_preds, all_targets

    def optimize_threshold(self, targets, probs):
        """
        Grid search for the threshold that maximizes MCC.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range 0.1 to 0.9
        thresholds = np.linspace(0.1, 0.9, 81)

        for t in thresholds:
            preds = (probs > t).astype(int)
            mcc = matthews_corrcoef(targets, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        return best_mcc, best_thresh

    def fit(self, epochs=Config.EPOCHS, debug=False):
        """
        Main training loop with Early Stopping.
        """
        # Reproducibility
        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        random.seed(Config.SEED)

        print(f"Initializing datasets (Debug={debug})...")
        train_ds = NFLContactDataset(split="train", debug=debug)
        val_ds = NFLContactDataset(split="validation", debug=debug)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        best_val_mcc = -1.0
        patience_counter = 0

        print("Starting training...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_probs, val_targets = self.validate(val_loader)

            val_mcc, val_thresh = self.optimize_threshold(val_targets, val_probs)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}: Train Loss {train_loss}, Val Loss {val_loss}, Val MCC {val_mcc} (Thresh {val_thresh})"
            )

            # Early Stopping and Checkpointing
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                print(f"New best MCC: {best_val_mcc}. Saving model...")
                torch.save(self.model.state_dict(), self.best_model_path)
                np.save(self.best_thresh_path, np.array([val_thresh]))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        return best_val_mcc

    def predict_and_submit(self):
        """
        Generates predictions on the test set using the best model and threshold.
        Saves to ./submission/submission.csv
        """
        # Load best model
        if not os.path.exists(self.best_model_path):
            print("No best model found. Cannot predict.")
            return

        print(f"Loading model from {self.best_model_path}...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Load threshold
        if os.path.exists(self.best_thresh_path):
            threshold = float(np.load(self.best_thresh_path))
        else:
            threshold = 0.5

        print(f"Using optimized threshold: {threshold}")

        print("Initializing Test Dataset...")
        test_ds = NFLContactDataset(split="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        contact_ids = []
        predictions = []

        print("Running Inference...")
        with torch.no_grad():
            for batch in test_loader:
                kin = batch["kinematic"].to(self.device)
                vis = batch["visual"].to(self.device)
                cat = batch["categorical"].to(self.device)
                c_ids = batch["contact_id"]

                logits = self.model(kin, vis, cat)
                probs = torch.sigmoid(logits)

                preds = (probs > threshold).int().cpu().numpy().flatten()

                contact_ids.extend(c_ids)
                predictions.extend(preds)

        # Save submission
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"

        df = pd.DataFrame({"contact_id": contact_ids, "contact": predictions})
        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path} with {len(df)} rows.")
