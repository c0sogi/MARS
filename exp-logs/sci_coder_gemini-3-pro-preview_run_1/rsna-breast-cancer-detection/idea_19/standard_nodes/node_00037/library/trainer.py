import os
import time
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

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
    VAL_METADATA_PATH,
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


def run_experiment_isolated(max_samples, epochs, metric_threshold):
    """
    Runs the full experiment pipeline in an isolated process to ensure resource cleanup.
    Logic moved from runfile.py to here to support multiprocessing spawn.
    Cite debug_lesson_9, debug_lesson_10.
    """
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    seed_everything(SEED)
    print(f"Configuration: Max Samples={max_samples}, Epochs={epochs}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, max_samples=max_samples
    )

    # =========================================================================
    # 3. Model & Trainer Initialization
    # =========================================================================
    print("Initializing Model and Trainer...")
    net = PyramidSiameseEfficientNet()
    t = Trainer(net, train_loader, val_loader, test_loader)

    # =========================================================================
    # 4. Training
    # =========================================================================
    print("Starting Training...")
    t.fit(epochs=epochs)

    # =========================================================================
    # 5. Validation & Metric Calculation
    # =========================================================================
    print("Performing Final Validation...")
    net.eval()

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for target_img, contra_img, labels, _ in val_loader:
            target_img = target_img.to(DEVICE)
            contra_img = contra_img.to(DEVICE)

            logits = net(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Probabilistic F1
    pf1 = probabilistic_f1(all_targets, all_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {pf1}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n==== Failure Analysis ====")

    # Load validation metadata to match the subset used
    df_val = pd.read_csv(VAL_METADATA_PATH)
    if max_samples:
        df_val = df_val.iloc[:max_samples]

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)
    df_val["error"] = errors

    # Encode Density for correlation (A->1, B->2, etc.)
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    df_val["density_enc"] = df_val["density"].map(density_map)

    # Compute Correlations
    features_to_analyze = ["age", "density_enc", "implant"]
    print("Correlation between Error Magnitude and Features:")

    for feat in features_to_analyze:
        if feat in df_val.columns:
            # Drop NaNs for this specific pair
            subset = df_val[[feat, "error"]].dropna()
            if len(subset) > 1:
                corr, _ = pearsonr(subset[feat], subset["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Feature not found")

    # =========================================================================
    # 7. Conditional Submission
    # =========================================================================
    print("\n==== Submission Generation ====")

    if pf1 > metric_threshold:
        print(
            f"Validation Metric ({pf1}) > Threshold ({metric_threshold}). Generating submission..."
        )

        # Run inference using the Trainer's method (loads best model)
        t.predict_and_submit()

        # Ensure file is saved to ./submission/submission.csv as requested
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        # The trainer saves to SUBMISSION_PATH (./submission.csv)
        if os.path.exists(SUBMISSION_PATH):
            shutil.copy(SUBMISSION_PATH, target_path)
            print(f"Submission file successfully saved to {target_path}")
        else:
            print("Error: Source submission file not found.")

    else:
        print(
            f"Validation Metric ({pf1}) <= Threshold ({metric_threshold}). Submission skipped."
        )
