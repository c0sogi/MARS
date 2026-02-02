import os
import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import JigsawModel
from library.dataset import ToxicityDataset, collate_batch
from library.data_utils import build_or_load_vocabulary, identify_identity_indices
from library.metrics import calculate_jigsaw_metrics


class Trainer:
    """
    Manages the training lifecycle for the Toxicity Classification model.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.tokenizer = None

    def set_seed(self, seed=Config.SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(
        self,
        train_df,
        val_df,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.PATIENCE,
        seed=Config.SEED,
    ):
        self.set_seed(seed)
        print(f"Starting training on device: {self.device}")

        # 1. Prepare Tokenizer and Identity Indices
        self.tokenizer = build_or_load_vocabulary(load_cached_data=True)
        identity_indices = identify_identity_indices(self.tokenizer)

        # 2. Calculate Sample Weights (Cite solution_lesson_node_00010)
        # Weight = 1.0 (base)
        # +0.25 if identity mentioned
        # +1.5 if identity mentioned AND non-toxic (target < 0.5)
        print("Calculating sample weights...")

        # Helper to get max identity value per row
        id_max = train_df[Config.IDENTITY_COLUMNS].max(axis=1).fillna(0.0)
        has_identity = id_max >= 0.5
        is_nontoxic = train_df[Config.TARGET_COL] < 0.5

        weights = np.ones(len(train_df))
        weights += has_identity * 0.25
        weights += (has_identity & is_nontoxic) * 1.5

        # 3. Prepare Datasets
        train_dataset = ToxicityDataset(
            texts=train_df[Config.TEXT_COL].tolist(),
            targets=train_df[Config.TARGET_COL].tolist(),
            aux_targets=train_df[Config.AUX_COLUMNS].values,
            weights=weights,
            tokenizer=self.tokenizer,
            identity_indices=identity_indices,
            mask_prob=Config.IDENTITY_MASK_PROB,
            is_training=True,
        )

        val_dataset = ToxicityDataset(
            texts=val_df[Config.TEXT_COL].tolist(),
            targets=val_df[Config.TARGET_COL].tolist(),
            aux_targets=val_df[Config.AUX_COLUMNS].values,
            weights=None,  # No weighting in validation
            tokenizer=self.tokenizer,
            identity_indices=identity_indices,
            mask_prob=0.0,
            is_training=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        # 4. Initialize Model (Cite solution_lesson_node_00006)
        # Output dim = 1 (main) + 6 (aux) = 7
        self.model = JigsawModel(num_labels=1 + len(Config.AUX_COLUMNS)).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # BCEWithLogitsLoss combines Sigmoid + BCELoss
        # reduction='none' to apply sample weights manually
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        # 5. Training Loop
        best_score = -float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()

            # --- Train Step ---
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                weights = batch["weights"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)

                # Calculate Loss (Cite solution_lesson_node_00008, solution_lesson_node_00010)
                # Main task is index 0, Aux tasks are indices 1:
                loss_per_sample = criterion(logits, targets)

                # Apply weights to main task loss
                main_loss = (loss_per_sample[:, 0] * weights).mean()

                # Aux loss (unweighted or standard weight 0.5)
                aux_loss = loss_per_sample[:, 1:].mean()

                total_loss = main_loss + 0.5 * aux_loss

                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item() * targets.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # --- Validation Step ---
            self.model.eval()
            val_loss = 0.0
            val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    targets = batch["targets"].to(self.device)

                    logits = self.model(input_ids, attention_mask)

                    # Validation loss (no weighting)
                    loss_per_sample = criterion(logits, targets)
                    main_loss = loss_per_sample[:, 0].mean()
                    aux_loss = loss_per_sample[:, 1:].mean()
                    total_loss = main_loss + 0.5 * aux_loss

                    val_loss += total_loss.item() * targets.size(0)

                    # Store predictions (sigmoid applied here for metric calculation)
                    probs = torch.sigmoid(logits[:, 0])
                    val_preds.extend(probs.cpu().numpy())

            avg_val_loss = val_loss / len(val_dataset)

            # --- Metric Calculation ---
            val_df_eval = val_df.copy()
            val_df_eval["prediction"] = val_preds

            metrics = calculate_jigsaw_metrics(val_df_eval, prediction_col="prediction")
            final_score = metrics["final_score"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"Score: {final_score:.10f} | "
                f"Overall AUC: {metrics['overall_auc']:.6f} | "
                f"Bias AUCs: {metrics['subgroup_auc_mean']:.4f}/{metrics['bpsn_auc_mean']:.4f}/{metrics['bnsp_auc_mean']:.4f}"
            )

            # --- Early Stopping ---
            if final_score > best_score:
                best_score = final_score
                # Cite solution_lesson_node_00002 (Deep Copy)
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Training complete. Best Score: {best_score}")

    def predict(self, test_df, batch_size=Config.BATCH_SIZE):
        if self.tokenizer is None:
            self.tokenizer = build_or_load_vocabulary(load_cached_data=True)

        if self.model is None:
            if os.path.exists(Config.MODEL_SAVE_PATH):
                print(f"Loading model from {Config.MODEL_SAVE_PATH}")
                # Re-init model
                self.model = JigsawModel(num_labels=1 + len(Config.AUX_COLUMNS)).to(
                    self.device
                )
                self.model.load_state_dict(
                    torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
                )
            else:
                raise ValueError("No trained model found.")

        self.model.eval()

        dataset = ToxicityDataset(
            texts=test_df[Config.TEXT_COL].tolist(),
            targets=None,
            tokenizer=self.tokenizer,
            is_training=False,
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits[:, 0])  # Main target only

                all_preds.extend(probs.cpu().numpy())

        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], "prediction": all_preds}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        return submission
