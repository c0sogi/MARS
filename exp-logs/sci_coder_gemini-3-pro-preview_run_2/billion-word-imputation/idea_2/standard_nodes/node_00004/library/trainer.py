import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

from library.config import Config
from library.model import BiLSTMDualHead
from library.dataset import get_dataloaders
from library.tokenizer import get_tokenizer


class Trainer:
    """
    Trainer class for the Bi-Directional LSTM Word Insertion Model.
    Handles training, validation, and submission generation.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Set seeds for reproducibility
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

        # Initialize DataLoaders and Tokenizer
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = (
            get_dataloaders(config=self.config, load_cached_data=True)
        )

        # Initialize Model
        print("Initializing Model...")
        self.model = BiLSTMDualHead(
            vocab_size=self.tokenizer.get_vocab_size(),
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            lstm_layers=self.config.LSTM_LAYERS,
            dropout=self.config.DROPOUT,
        ).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Location Loss: Binary classification per token
        self.loc_criterion = nn.BCEWithLogitsLoss()

        # Word Generation Loss: Multi-class classification
        self.word_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch_idx):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        running_loc_loss = 0.0
        running_word_loss = 0.0

        # Use tqdm for progress tracking if desired, but prompt says avoid progress bars.
        # We will iterate silently or print periodic updates if needed.

        for batch in self.train_loader:
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            loc_target = batch["loc_target"].to(self.device)
            word_target = batch["word_target"].to(self.device)
            gap_idx = batch["gap_idx"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # loc_logits: (batch, seq_len, 1)
            # word_logits: (batch, seq_len, vocab_size)
            loc_logits, word_logits = self.model(input_ids)

            # 1. Location Loss
            # Squeeze loc_logits to match loc_target shape (batch, seq_len)
            loc_loss = self.loc_criterion(loc_logits.squeeze(-1), loc_target)

            # 2. Word Generation Loss
            # We only care about the prediction at the specific gap index.
            # We need to gather the logits corresponding to the gap_idx for each sample in the batch.

            # Create batch indices [0, 1, ..., batch_size-1]
            batch_indices = torch.arange(input_ids.size(0), device=self.device)

            # Select logits at the gap position: (batch, vocab_size)
            target_word_logits = word_logits[batch_indices, gap_idx, :]

            word_loss = self.word_criterion(target_word_logits, word_target)

            # Combined Loss
            total_loss = loc_loss + (self.config.LOSS_LAMBDA * word_loss)

            # Backward pass
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.CLIP_GRAD
            )

            self.optimizer.step()

            running_loss += total_loss.item()
            running_loc_loss += loc_loss.item()
            running_word_loss += word_loss.item()

        num_batches = len(self.train_loader)
        avg_loss = running_loss / num_batches
        avg_loc = running_loc_loss / num_batches
        avg_word = running_word_loss / num_batches

        return avg_loss, avg_loc, avg_word

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct_loc = 0
        correct_word = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                loc_target = batch["loc_target"].to(self.device)
                word_target = batch["word_target"].to(self.device)
                gap_idx = batch["gap_idx"].to(self.device)

                loc_logits, word_logits = self.model(input_ids)

                # Loss Calculation (Same as train)
                loc_loss = self.loc_criterion(loc_logits.squeeze(-1), loc_target)
                batch_indices = torch.arange(input_ids.size(0), device=self.device)
                target_word_logits = word_logits[batch_indices, gap_idx, :]
                word_loss = self.word_criterion(target_word_logits, word_target)
                total_loss = loc_loss + (self.config.LOSS_LAMBDA * word_loss)

                running_loss += total_loss.item()

                # Metrics Calculation
                # 1. Location Accuracy: Did we predict the highest prob at the correct gap_idx?
                # loc_logits shape: (batch, seq_len, 1) -> (batch, seq_len)
                pred_loc_idx = torch.argmax(loc_logits.squeeze(-1), dim=1)
                correct_loc += (pred_loc_idx == gap_idx).sum().item()

                # 2. Word Accuracy: Did we predict the correct word at the correct gap?
                # We evaluate word accuracy assuming we are looking at the ground truth gap location
                # to isolate generation performance.
                pred_word_idx = torch.argmax(target_word_logits, dim=1)
                correct_word += (pred_word_idx == word_target).sum().item()

                total_samples += input_ids.size(0)

        avg_loss = running_loss / len(self.val_loader)
        acc_loc = correct_loc / total_samples
        acc_word = correct_word / total_samples

        return avg_loss, acc_loc, acc_word

    def fit(self):
        """Main training loop with Early Stopping."""
        print(f"Starting training on device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            # Train
            train_loss, train_loc, train_word = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss} (Loc: {train_loc}, Word: {train_word})")

            # Validate
            val_loss, val_acc_loc, val_acc_word = self.evaluate()
            print(f"Val Loss: {val_loss}")
            print(f"Val Loc Acc: {val_acc_loc}")
            print(f"Val Word Acc: {val_acc_word}")

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"New best model saved to {self.config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )
                if patience_counter >= self.config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("\nGenerating submission...")

        # Load best model
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()

        results = []

        # Ensure submission directory exists
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

        with torch.no_grad():
            for batch in self.test_loader:
                ids = batch["id"].numpy()
                input_ids = batch["input_ids"].to(self.device)

                # Forward pass
                loc_logits, word_logits = self.model(input_ids)

                # Get Predictions
                # 1. Find best location index for each sample
                # Shape: (batch, seq_len)
                loc_probs = torch.sigmoid(loc_logits.squeeze(-1))

                # Mask out padding tokens to prevent predicting gap after padding
                # Pad token ID is 0.
                padding_mask = input_ids != self.tokenizer.pad_token_id
                loc_probs = loc_probs * padding_mask.float()

                best_loc_indices = torch.argmax(loc_probs, dim=1)  # Shape: (batch,)

                # 2. Find best word at that location
                batch_indices = torch.arange(input_ids.size(0), device=self.device)

                # Gather word logits at the predicted location
                # Shape: (batch, vocab_size)
                best_word_logits = word_logits[batch_indices, best_loc_indices, :]

                pred_word_ids = torch.argmax(best_word_logits, dim=1)  # Shape: (batch,)

                # Move to CPU for processing
                input_ids_np = input_ids.cpu().numpy()
                best_loc_indices_np = best_loc_indices.cpu().numpy()
                pred_word_ids_np = pred_word_ids.cpu().numpy()

                for i in range(len(ids)):
                    sample_id = ids[i]
                    curr_input_ids = input_ids_np[i]
                    gap_idx = best_loc_indices_np[i]
                    pred_word_id = pred_word_ids_np[i]

                    # Decode predicted word
                    pred_word = self.tokenizer.idx2word.get(
                        pred_word_id, self.tokenizer.unk_token
                    )

                    # Reconstruct Sentence
                    # The gap_idx represents the token *before* the missing word.
                    # So we insert the predicted word after gap_idx.

                    # Convert input ids to words (excluding padding)
                    tokens = []
                    for tid in curr_input_ids:
                        if tid == self.tokenizer.pad_token_id:
                            break
                        tokens.append(
                            self.tokenizer.idx2word.get(tid, self.tokenizer.unk_token)
                        )

                    # Insert word
                    # If gap_idx is 0, it means insert after the first token (index 0).
                    # Python list insert: index is the position *before* which to insert.
                    # So to insert after index 0, we insert at index 1.
                    # insert_pos = gap_idx + 1

                    # Safety check for bounds
                    insert_pos = min(gap_idx + 1, len(tokens))
                    tokens.insert(insert_pos, pred_word)

                    final_sentence = " ".join(tokens)

                    # Escape quotes for CSV format as per requirement
                    # "Use double quotes to escape the sentence text and two double quotes ("") for double quotes within a sentence."
                    # The pandas to_csv with quoting=csv.QUOTE_NONNUMERIC handles the outer quotes.
                    # We just need to ensure internal quotes are escaped if we were writing manually,
                    # but pandas handles standard CSV escaping.

                    results.append({"id": sample_id, "sentence": final_sentence})

        # Save to CSV
        df_sub = pd.DataFrame(results)

        # The submission format requires: id,"sentence"
        # Pandas to_csv handles this well.
        df_sub.to_csv(
            self.config.SUBMISSION_PATH,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,  # Ensures non-numeric fields (sentence) are quoted
        )
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def run_training_pipeline():
    """Helper to run the full pipeline."""
    trainer = Trainer()
    trainer.fit()
    trainer.generate_submission()
