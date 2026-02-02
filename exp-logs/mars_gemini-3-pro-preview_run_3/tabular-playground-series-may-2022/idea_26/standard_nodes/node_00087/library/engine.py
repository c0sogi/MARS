import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.dataset import ManufacturingDataset
from library.model import RPFEModel


class Engine:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.set_seed(Config.SEED)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_fn(self, model, dataloader, optimizer, scheduler, criterion):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            continuous = batch["continuous"].to(self.device)
            categorical = batch["categorical"].to(self.device)
            targets = batch["target"].to(self.device)

            optimizer.zero_grad()

            # Forward pass: returns (batch_size, num_streams)
            outputs = model(continuous, categorical)

            # Loss calculation: Sum of BCE losses for each stream
            loss = 0.0
            for i in range(Config.NUM_STREAMS):
                # Slice output for stream i: (batch_size, 1)
                stream_out = outputs[:, i : i + 1]
                loss += criterion(stream_out, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / num_batches

    def eval_fn(self, model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = model(continuous, categorical)

                # Average probabilities across streams
                probs = torch.sigmoid(outputs).mean(dim=1)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets).ravel()

        return roc_auc_score(all_targets, all_preds)

    def predict_fn(self, model, dataloader):
        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)

                outputs = model(continuous, categorical)

                # Average probabilities across streams
                probs = torch.sigmoid(outputs).mean(dim=1)

                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds)

    def run(self, debug_sample_size=None, epochs=Config.EPOCHS):
        print(f"Starting execution on device: {self.device}")

        # 1. Data Processing
        fe = FeatureEngineer()
        train_df, val_df, test_df, vocab_sizes = fe.process_data(
            load_cached_data=True, debug_sample_size=debug_sample_size
        )

        # Identify columns
        # Categorical columns are f_27_0 through f_27_9
        cat_cols = [f"f_27_{i}" for i in range(10)]

        # Continuous columns are all others except metadata
        exclude_cols = {"id", "target", "source_path"} | set(cat_cols)
        cont_cols = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Continuous features: {len(cont_cols)}")
        print(f"Categorical features: {len(cat_cols)}")

        # 2. Datasets & Loaders
        train_ds = ManufacturingDataset(train_df, cat_cols, cont_cols, is_test=False)
        val_ds = ManufacturingDataset(val_df, cat_cols, cont_cols, is_test=False)
        test_ds = ManufacturingDataset(test_df, cat_cols, cont_cols, is_test=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Model Initialization
        model = RPFEModel(vocab_sizes, len(cont_cols))
        model.to(self.device)

        # 4. Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        criterion = nn.BCEWithLogitsLoss()

        # 5. Training Loop
        best_auc = 0.0
        patience_counter = 0

        print("Starting training...")
        for epoch in range(epochs):
            train_loss = self.train_fn(
                model, train_loader, optimizer, scheduler, criterion
            )
            val_auc = self.eval_fn(model, val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # 6. Inference
        print("Loading best model for inference...")
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )

        predictions = self.predict_fn(model, test_loader)

        # 7. Submission
        # Load sample submission to get IDs
        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Handle potential length mismatch if debugging
        if len(predictions) != len(submission):
            print(
                f"Adjusting submission length from {len(submission)} to {len(predictions)} (Debug Mode)"
            )
            submission = submission.iloc[: len(predictions)].copy()

        submission["target"] = predictions
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
