import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, compute_kendall_tau
from library.data_processing import load_data, NotebookDataset
from library.feature_engineering import create_sparse_features
from library.models import TransformerRanker, RidgeRanker


class Trainer:
    """
    Trainer class to manage the training, validation, and submission generation
    for the Notebook Cell Ordering task.
    """

    def __init__(self, debug=False):
        """
        Initializes the Trainer.

        Args:
            debug (bool): If True, uses a smaller subset of data for debugging.
        """
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        print(f"Initializing Trainer (Debug={self.debug}, Device={self.device})...")

        # ----------------------------------------------------------------------
        # 1. Data Loading
        # ----------------------------------------------------------------------
        # Load processed dataframes (cached or fresh)
        self.train_df = load_data("train", debug=self.debug)
        self.val_df = load_data("val", debug=self.debug)

        # Load Metadata for Validation Reconstruction
        self.val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)
        if self.debug:
            # Filter metadata to match the debug dataframe
            valid_ids = self.val_df["id"].unique()
            self.val_metadata = self.val_metadata[
                self.val_metadata["id"].isin(valid_ids)
            ]

        # ----------------------------------------------------------------------
        # 2. Feature Engineering
        # ----------------------------------------------------------------------
        # Generate/Load Sparse Features for Ridge Regression
        self.train_sparse = create_sparse_features(self.train_df, "train")
        self.val_sparse = create_sparse_features(self.val_df, "val")

        # ----------------------------------------------------------------------
        # 3. Dataset & Dataloader Setup (Dense Stream)
        # ----------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        self.train_dataset = NotebookDataset(
            self.train_df, self.tokenizer, max_len=Config.MAX_LEN
        )
        self.val_dataset = NotebookDataset(
            self.val_df, self.tokenizer, max_len=Config.MAX_LEN
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # ----------------------------------------------------------------------
        # 4. Model Setup
        # ----------------------------------------------------------------------
        # Sparse Model
        self.ridge_model = RidgeRanker()

        # Dense Model
        self.transformer_model = TransformerRanker(model_name=Config.MODEL_NAME)
        self.transformer_model.to(self.device)

        # Optimization
        self.optimizer = AdamW(
            self.transformer_model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.criterion = nn.MSELoss()

        # Scheduler
        num_train_steps = len(self.train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

    def train_ridge(self):
        """Trains the Ridge Regression model on sparse features."""
        print("\n--- Training Ridge Model ---")
        y_train = self.train_df["rank"].values
        self.ridge_model.fit(self.train_sparse, y_train)
        self.ridge_model.save()
        print("Ridge Model trained and saved.")

    def train_transformer_epoch(self, epoch_idx):
        """Runs one epoch of training for the Transformer model."""
        self.transformer_model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        # Using tqdm for visual feedback
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1} Train", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.transformer_model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch_idx+1} | Train MSE: {avg_loss:.6f}")

    def validate(self):
        """
        Evaluates the ensemble on the validation set.
        Computes the Kendall Tau score by reconstructing notebook orders.
        """
        print("\n--- Running Validation ---")
        self.transformer_model.eval()

        # 1. Get Transformer Predictions
        trans_preds = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val Inference (Transformer)"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.transformer_model(input_ids, attention_mask)
                trans_preds.append(outputs.cpu().numpy())
        trans_preds = np.concatenate(trans_preds)

        # 2. Get Ridge Predictions
        print("Val Inference (Ridge)...")
        ridge_preds = self.ridge_model.predict(self.val_sparse)

        # 3. Ensemble
        alpha = Config.ENSEMBLE_ALPHA
        final_preds = alpha * ridge_preds + (1 - alpha) * trans_preds

        # 4. Reconstruct Orders and Compute Metric
        # Map predictions back to (notebook_id, cell_id)
        val_df_pred = self.val_df.copy()
        val_df_pred["pred_rank"] = final_preds

        # Create a lookup: notebook_id -> {cell_id: pred_rank}
        pred_lookup = {}
        for nid, group in val_df_pred.groupby("id"):
            pred_lookup[nid] = dict(zip(group["cell_id"], group["pred_rank"]))

        submission_data = []

        # Iterate through validation notebooks to reconstruct order
        for _, row in tqdm(
            self.val_metadata.iterrows(),
            total=len(self.val_metadata),
            desc="Reconstructing Orders",
        ):
            nb_id = row["id"]
            if nb_id not in pred_lookup:
                continue

            filepath = row["filepath"]
            full_path = os.path.join(Config.INPUT_DIR, filepath)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception:
                continue

            # Identify code cells and assign anchor ranks
            cell_types = nb_json.get("cell_type", {})
            code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
            n_code = len(code_cells)

            cells_with_ranks = []

            # Anchor Logic: Equidistant ranks for code cells
            if n_code == 1:
                # Place single code cell in the middle
                cells_with_ranks.append((code_cells[0], 0.5))
            elif n_code > 1:
                for i, cid in enumerate(code_cells):
                    r = i / (n_code - 1)
                    cells_with_ranks.append((cid, r))

            # Add predicted markdown ranks
            md_preds = pred_lookup.get(nb_id, {})
            for cid, rank in md_preds.items():
                cells_with_ranks.append((cid, rank))

            # Sort by rank
            cells_with_ranks.sort(key=lambda x: x[1])

            # Form order string
            sorted_order = " ".join([x[0] for x in cells_with_ranks])
            submission_data.append({"id": nb_id, "cell_order": sorted_order})

        df_pred = pd.DataFrame(submission_data)
        df_gt = self.val_metadata[["id", "cell_order"]]

        # Compute Metric
        score = compute_kendall_tau(df_pred, df_gt)
        print(f"Validation Kendall Tau: {score:.6f}")
        return score

    def fit(self):
        """
        Main training loop.
        Trains Ridge, then Transformer with validation and checkpointing.
        """
        # Train Sparse Stream
        self.train_ridge()

        # Train Dense Stream
        best_score = -float("inf")

        print("\n--- Starting Transformer Training ---")
        for epoch in range(Config.EPOCHS):
            self.train_transformer_epoch(epoch)
            score = self.validate()

            if score > best_score:
                print(
                    f"Score improved ({best_score:.6f} -> {score:.6f}). Saving model..."
                )
                best_score = score
                self.transformer_model.save()
            else:
                print(f"Score did not improve (Best: {best_score:.6f}).")

        print(f"\nTraining Complete. Best Validation Score: {best_score:.6f}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("\n--- Generating Submission ---")

        # 1. Load Test Data
        test_df = load_data("test", debug=self.debug)
        test_sparse = create_sparse_features(test_df, "test")

        test_dataset = NotebookDataset(test_df, self.tokenizer, max_len=Config.MAX_LEN)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # 2. Load Best Models
        print("Loading best models...")
        self.transformer_model.load()
        self.ridge_model.load()
        self.transformer_model.eval()

        # 3. Predict
        trans_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Inference (Transformer)"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.transformer_model(input_ids, attention_mask)
                trans_preds.append(outputs.cpu().numpy())
        trans_preds = np.concatenate(trans_preds)

        print("Test Inference (Ridge)...")
        ridge_preds = self.ridge_model.predict(test_sparse)

        # Ensemble
        alpha = Config.ENSEMBLE_ALPHA
        final_preds = alpha * ridge_preds + (1 - alpha) * trans_preds

        test_df["pred_rank"] = final_preds

        # 4. Reconstruct Orders
        test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        if self.debug:
            test_metadata = test_metadata.head(len(test_df))

        # Group predictions
        pred_lookup = {}
        for nid, group in test_df.groupby("id"):
            pred_lookup[nid] = dict(zip(group["cell_id"], group["pred_rank"]))

        submission_data = []

        for _, row in tqdm(
            test_metadata.iterrows(),
            total=len(test_metadata),
            desc="Reconstructing Test Orders",
        ):
            nb_id = row["id"]
            filepath = row["filepath"]
            full_path = os.path.join(Config.INPUT_DIR, filepath)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception:
                submission_data.append({"id": nb_id, "cell_order": ""})
                continue

            cell_types = nb_json.get("cell_type", {})
            code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
            n_code = len(code_cells)

            cells_with_ranks = []

            # Anchor Logic
            if n_code == 1:
                cells_with_ranks.append((code_cells[0], 0.5))
            elif n_code > 1:
                for i, cid in enumerate(code_cells):
                    r = i / (n_code - 1)
                    cells_with_ranks.append((cid, r))

            # Add Predictions
            md_preds = pred_lookup.get(nb_id, {})
            for cid, rank in md_preds.items():
                cells_with_ranks.append((cid, rank))

            # Sort
            cells_with_ranks.sort(key=lambda x: x[1])
            sorted_order = " ".join([x[0] for x in cells_with_ranks])
            submission_data.append({"id": nb_id, "cell_order": sorted_order})

        # 5. Save
        df_sub = pd.DataFrame(submission_data)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
