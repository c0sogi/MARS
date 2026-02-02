import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import NBOWModel
from library.dataset import ToxicityDataset, collate_batch
from library.data_utils import build_or_load_vocabulary, identify_identity_indices
from library.metrics import calculate_jigsaw_metrics


class Trainer:
    """
    Manages the training lifecycle for the Toxicity Classification model.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (torch.device, optional): Device to run the model on.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.vocab = None

    def set_seed(self, seed=Config.SEED):
        """
        Sets random seeds for reproducibility.
        """
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
        """
        Executes the training loop with early stopping based on the competition metric.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            batch_size (int): Batch size for DataLoader.
            epochs (int): Maximum number of epochs.
            lr (float): Learning rate.
            patience (int): Early stopping patience.
            seed (int): Random seed.
        """
        self.set_seed(seed)
        print(f"Starting training on device: {self.device}")

        # 1. Prepare Vocabulary and Identity Indices
        # Utilizes caching mechanism in data_utils
        self.vocab = build_or_load_vocabulary(load_cached_data=True)
        identity_indices = identify_identity_indices(self.vocab)

        # 2. Prepare Datasets
        train_dataset = ToxicityDataset(
            texts=train_df[Config.TEXT_COL].tolist(),
            targets=train_df[Config.TARGET_COL].tolist(),
            vocab=self.vocab,
            identity_indices=identity_indices,
            mask_prob=Config.IDENTITY_MASK_PROB,
            is_training=True,
        )

        val_dataset = ToxicityDataset(
            texts=val_df[Config.TEXT_COL].tolist(),
            targets=val_df[Config.TARGET_COL].tolist(),
            vocab=self.vocab,
            identity_indices=identity_indices,
            mask_prob=0.0,  # No masking during validation
            is_training=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        # 3. Initialize Model
        self.model = NBOWModel(
            vocab_size=len(self.vocab),
            embed_dim=Config.EMBED_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout_rate=Config.DROPOUT,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 4. Training Loop
        best_score = -float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()

            # --- Train Step ---
            self.model.train()
            train_loss = 0.0

            for texts, offsets, targets in train_loader:
                texts = texts.to(self.device)
                offsets = offsets.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(texts, offsets).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * targets.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # --- Validation Step ---
            self.model.eval()
            val_loss = 0.0
            val_preds = []

            with torch.no_grad():
                for texts, offsets, targets in val_loader:
                    texts = texts.to(self.device)
                    offsets = offsets.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(texts, offsets).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * targets.size(0)

                    # Handle case where batch size is 1 and squeeze removed the dimension
                    if outputs.ndim == 0:
                        outputs = outputs.unsqueeze(0)

                    val_preds.extend(outputs.cpu().numpy())

            avg_val_loss = val_loss / len(val_dataset)

            # --- Metric Calculation ---
            # Create a temp dataframe with predictions to use the metric library
            val_df_eval = val_df.copy()
            val_df_eval["prediction"] = val_preds

            metrics = calculate_jigsaw_metrics(val_df_eval, prediction_col="prediction")
            final_score = metrics["final_score"]

            elapsed = time.time() - start_time

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss} | "
                f"Score: {final_score} | "
                f"Overall AUC: {metrics['overall_auc']} | "
                f"Bias AUCs (Sub/BPSN/BNSP): {metrics['subgroup_auc_mean']}/{metrics['bpsn_auc_mean']}/{metrics['bnsp_auc_mean']}"
            )

            # --- Early Stopping ---
            if final_score > best_score:
                best_score = final_score
                best_model_state = self.model.state_dict()
                patience_counter = 0
                # Save checkpoint
                os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Training complete. Best Score: {best_score}")

    def predict(self, test_df, batch_size=Config.BATCH_SIZE):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_df (pd.DataFrame): Test data.
            batch_size (int): Batch size for inference.

        Returns:
            pd.DataFrame: The submission dataframe.
        """
        # Ensure model and vocab are loaded
        if self.vocab is None:
            self.vocab = build_or_load_vocabulary(load_cached_data=True)

        if self.model is None:
            if os.path.exists(Config.MODEL_SAVE_PATH):
                print(f"Loading model from {Config.MODEL_SAVE_PATH}")
                self.model = NBOWModel(
                    vocab_size=len(self.vocab),
                    embed_dim=Config.EMBED_DIM,
                    hidden_dim=Config.HIDDEN_DIM,
                    dropout_rate=Config.DROPOUT,
                ).to(self.device)
                self.model.load_state_dict(
                    torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
                )
            else:
                raise ValueError("No trained model found in memory or on disk.")

        self.model.eval()

        dataset = ToxicityDataset(
            texts=test_df[Config.TEXT_COL].tolist(),
            targets=None,
            vocab=self.vocab,
            is_training=False,
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )

        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in loader:
                # collate_batch returns (texts, offsets) when targets are None
                texts, offsets = batch
                texts = texts.to(self.device)
                offsets = offsets.to(self.device)

                outputs = self.model(texts, offsets).squeeze()

                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)

                all_preds.extend(outputs.cpu().numpy())

        # Create submission file
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], "prediction": all_preds}
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        return submission
