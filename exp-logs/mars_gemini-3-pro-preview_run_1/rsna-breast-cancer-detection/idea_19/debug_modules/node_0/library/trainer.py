import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    POS_WEIGHT,
    SCHEDULER,
    T_MAX,
    SUBMISSION_PATH,
    WORKING_DIR,
    CACHE_DIR,
    SEED,
)
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import PyramidSiameseEfficientNet


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss Function with Aggressive Positive Weighting
        # pos_weight must be a tensor on the same device
        pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_MAX
        )

        self.best_val_loss = float("inf")
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        # No progress bar printed as per requirements, just logic
        for batch_idx, (target_img, contra_img, labels, _) in enumerate(
            self.train_loader
        ):
            target_img = target_img.to(DEVICE)
            contra_img = contra_img.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(target_img, contra_img)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # NOTE: Gradient Clipping is explicitly DISABLED per instructions
            # to allow large updates for the minority class.

            self.optimizer.step()

            # Metrics tracking
            running_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

        avg_loss = running_loss / len(self.train_loader)
        pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

        return avg_loss, pf1

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for target_img, contra_img, labels, _ in self.val_loader:
                target_img = target_img.to(DEVICE)
                contra_img = contra_img.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                logits = self.model(target_img, contra_img)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()

                all_preds.extend(probs)
                all_targets.extend(targets)

        avg_loss = running_loss / len(self.val_loader)
        pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

        return avg_loss, pf1

    def fit(self, epochs=EPOCHS):
        print(f"Starting training for {epochs} epochs on {DEVICE}...")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_pf1 = self.train_one_epoch(epoch)
            val_loss, val_pf1 = self.validate()

            self.scheduler.step()

            elapsed = time.time() - start_time

            print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
            print(f"  Train Loss: {train_loss} | Train pF1: {train_pf1}")
            print(f"  Val Loss:   {val_loss} | Val pF1:   {val_pf1}")

            # Checkpointing (Minimize Val Loss)
            if val_loss < self.best_val_loss:
                print(
                    f"  [Improvement] Val Loss decreased from {self.best_val_loss} to {val_loss}. Saving model."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

            print("-" * 30)

    def predict_and_submit(self):
        print("Loading best model for inference...")
        if not os.path.exists(self.best_model_path):
            print("Warning: Best model not found. Using current weights.")
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=DEVICE)
            )

        self.model.eval()

        prediction_ids = []
        probabilities = []

        print("Running inference on test set...")
        with torch.no_grad():
            for target_img, contra_img, _, pred_ids in self.test_loader:
                target_img = target_img.to(DEVICE)
                contra_img = contra_img.to(DEVICE)

                logits = self.model(target_img, contra_img)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                prediction_ids.extend(pred_ids)
                probabilities.extend(probs)

        # Create DataFrame
        df_pred = pd.DataFrame(
            {"prediction_id": prediction_ids, "cancer": probabilities}
        )

        # Aggregation Strategy: Max Probability per prediction_id
        # Multiple images (views) map to the same prediction_id (e.g. breast).
        # We take the maximum likelihood of cancer among all views for that breast.
        df_submission = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

        # Save submission
        print(f"Saving submission to {SUBMISSION_PATH}...")
        df_submission.to_csv(SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
        print(df_submission.head())


def run_training_pipeline(load_cached_data=True, max_samples=None, epochs=EPOCHS):
    """
    Main entry point to run the training and submission pipeline.

    Args:
        load_cached_data (bool): Whether to use cached stats/metadata.
        max_samples (int): Optional limit on dataset size for debugging.
        epochs (int): Number of training epochs.
    """
    # 1. Reproducibility
    seed_everything(SEED)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, max_samples=max_samples
    )

    # 3. Model Initialization
    print("Initializing Pyramid Siamese Network...")
    model = PyramidSiameseEfficientNet()

    # 4. Trainer Setup
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # 5. Training
    trainer.fit(epochs=epochs)

    # 6. Inference & Submission
    trainer.predict_and_submit()
