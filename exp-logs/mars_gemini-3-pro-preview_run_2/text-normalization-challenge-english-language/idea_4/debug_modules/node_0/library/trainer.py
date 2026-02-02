import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from library.config import Config
from library.model import TokenClassifier, train_epoch, evaluate, predict_labels
from library.dataset import TextNormalizationDataset
from library.normalization_rules import Normalizer


class Trainer:
    """
    Manages the training, validation, and submission generation loops for the
    Text Normalization model.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = TokenClassifier()
        self.model.to(self.device)

    def train_epoch(self, dataloader, optimizer, scheduler):
        """
        Wrapper for the library.model.train_epoch function.
        """
        return train_epoch(self.model, dataloader, optimizer, scheduler, self.device)

    def evaluate(self, dataloader):
        """
        Wrapper for the library.model.evaluate function.
        """
        return evaluate(self.model, dataloader, self.device)

    def train(self, train_dataset, val_dataset, epochs=Config.EPOCHS):
        """
        Runs the main training loop with Early Stopping.
        """
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Optimizer and Scheduler
        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_val_loss = float("inf")
        patience = 2
        patience_counter = 0

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.evaluate(val_loader)

            # Print full precision metrics
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print("Saved best model.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model for inference
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )

    def generate_submission(self, test_dataset):
        """
        Generates predictions for the test set, applies normalization rules,
        and saves the submission file.
        """
        print("Generating predictions for test set...")

        # Get predictions (list of lists of labels)
        preds_by_sentence = predict_labels(self.model, test_dataset)

        # Flatten predictions to match the token-level test dataframe
        flat_preds = [label for sent in preds_by_sentence for label in sent]

        # Load test metadata to align predictions with raw text
        # We must ensure we load it exactly as dataset.py does to ensure order
        print("Loading test metadata for alignment...")
        df_test = pd.read_csv(
            Config.TEST_DATA_PATH,
            keep_default_na=False,
            dtype={"sentence_id": int, "token_id": int},
        )

        # Ensure correct order (dataset.py sorts by sentence_id, token_id)
        df_test = df_test.sort_values(["sentence_id", "token_id"])

        if len(flat_preds) != len(df_test):
            print(
                f"Warning: Prediction count {len(flat_preds)} != Test Data count {len(df_test)}"
            )

        # Assign predicted classes
        df_test["class"] = flat_preds

        # Apply Normalization Rules
        print("Applying deterministic normalization rules...")
        norm = Normalizer()

        # Apply row-wise normalization based on predicted class
        # Note: 'before' column contains raw text
        df_test["after"] = df_test.apply(
            lambda row: norm.normalize(row["before"], row["class"]), axis=1
        )

        # Format for submission
        submission = df_test[["id", "after"]]

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generation complete.")

    def run(self, debug=False, epochs=Config.EPOCHS):
        """
        Orchestrates the full pipeline: Data Loading -> Training -> Inference.
        """
        debug_size = 1000 if debug else None

        print(f"Initializing datasets (Debug={debug})...")
        train_ds = TextNormalizationDataset(
            split="train", load_cached_data=True, debug_size=debug_size
        )
        val_ds = TextNormalizationDataset(
            split="val", load_cached_data=True, debug_size=debug_size
        )

        # Run Training
        self.train(train_ds, val_ds, epochs=epochs)

        # Run Inference
        test_ds = TextNormalizationDataset(
            split="test", load_cached_data=True, debug_size=debug_size
        )
        self.generate_submission(test_ds)
