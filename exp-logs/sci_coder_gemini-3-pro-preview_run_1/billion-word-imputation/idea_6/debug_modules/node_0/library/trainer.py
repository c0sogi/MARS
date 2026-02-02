import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.vocab import Vocabulary
from library.data import InterleavedDataset, collate_fn
from library.model import BifurcatedTransformer


class Trainer:
    """
    Trainer for the Bifurcated Interleaved Transformer (Idea 6).
    Manages training, validation, and submission generation.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        self.set_seed(Config.SEED)

        # Initialize Vocabulary
        self.vocab = Vocabulary()
        self.vocab.build(load_cached_data=True)

        # Initialize Model
        self.model = BifurcatedTransformer().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = None  # Initialized in train()

        # Loss Functions
        # Localization: Binary Cross Entropy with Logits
        # We use a positive weight to account for the imbalance (1 gap vs many non-gaps)
        # Assuming avg seq len ~25, pos_weight ~20 is a reasonable heuristic.
        self.loc_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([20.0]).to(self.device), reduction="none"
        )

        # Identification: Cross Entropy
        self.id_criterion = nn.CrossEntropyLoss()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random

        random.seed(seed)

    def train(self, epochs=Config.EPOCHS):
        print(f"Starting training on {self.device}...")

        # DataLoaders
        # Debug mode limits samples for rapid testing
        max_samples = 1000 if self.debug else None

        train_dataset = InterleavedDataset(
            "train", self.vocab, load_cached_data=True, max_samples=max_samples
        )
        val_dataset = InterleavedDataset(
            "val", self.vocab, load_cached_data=True, max_samples=max_samples
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            train_loc_acc = 0.0
            train_id_acc = 0.0
            steps = 0

            for batch in train_loader:
                loss, loc_acc, id_acc = self._step(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()
                train_loc_acc += loc_acc
                train_id_acc += id_acc
                steps += 1

            avg_train_loss = train_loss / steps
            avg_train_loc_acc = train_loc_acc / steps
            avg_train_id_acc = train_id_acc / steps

            # Validation Phase
            val_loss, val_loc_acc, val_id_acc = self.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs}")
            print(
                f"Train Loss: {avg_train_loss} | Loc Acc: {avg_train_loc_acc} | ID Acc: {avg_train_id_acc}"
            )
            print(
                f"Val Loss: {val_loss} | Loc Acc: {val_loc_acc} | ID Acc: {val_id_acc}"
            )

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"Saved best model to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def _step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_loc = batch["target_loc"].to(self.device)  # (B,)
        target_word = batch["target_word"].to(self.device)  # (B,)

        # Forward Pass
        # loc_logits: (B, S, 1)
        # id_logits: (B, S, V)
        loc_logits, id_logits = self.model(input_ids, attention_mask)

        # --- Localization Loss ---
        # Identify valid samples (target_loc != -1)
        valid_mask = target_loc >= 0
        batch_size, seq_len = input_ids.shape

        # Create binary target mask for BCE
        loc_targets = torch.zeros((batch_size, seq_len), device=self.device)

        # Handle scatter indices: clamp -1 to 0 temporarily, then mask later
        safe_target_loc = target_loc.clone()
        safe_target_loc[~valid_mask] = 0
        loc_targets.scatter_(1, safe_target_loc.unsqueeze(1), 1.0)

        loc_logits_flat = loc_logits.squeeze(-1)  # (B, S)

        # Calculate BCE (unreduced)
        bce_loss = self.loc_criterion(loc_logits_flat, loc_targets)

        # Apply Masks:
        # 1. Attention mask (ignore padding tokens)
        # 2. Valid mask (ignore samples where target was truncated)
        loss_mask = attention_mask.float() * valid_mask.unsqueeze(1).float()

        loc_loss = (bce_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        # --- Identification Loss ---
        # Only compute loss at the specific gap location

        # Filter for valid samples
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)

        if valid_indices.numel() > 0:
            # Extract valid data
            v_target_loc = target_loc[valid_indices]  # (B_valid,)
            v_target_word = target_word[valid_indices]  # (B_valid,)
            v_id_logits = id_logits[valid_indices]  # (B_valid, S, V)

            # Gather logits at the gap index
            # Index shape must match v_id_logits except at dim 1
            gather_index = v_target_loc.view(-1, 1, 1).expand(-1, 1, Config.VOCAB_SIZE)
            selected_logits = torch.gather(v_id_logits, 1, gather_index).squeeze(
                1
            )  # (B_valid, V)

            id_loss = self.id_criterion(selected_logits, v_target_word)
        else:
            id_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Total Loss
        total_loss = Config.LAMBDA_LOC * loc_loss + Config.LAMBDA_ID * id_loss

        # --- Metrics ---
        with torch.no_grad():
            # Localization Accuracy
            # Mask padding in logits before argmax
            masked_loc_logits = loc_logits_flat.clone()
            masked_loc_logits[attention_mask == 0] = -float("inf")
            pred_loc = torch.argmax(masked_loc_logits, dim=1)

            loc_correct = (pred_loc == target_loc) & valid_mask
            loc_acc = loc_correct.float().sum() / (valid_mask.float().sum() + 1e-8)

            # Identification Accuracy
            if valid_indices.numel() > 0:
                pred_word = torch.argmax(selected_logits, dim=1)
                id_correct = pred_word == v_target_word
                id_acc = id_correct.float().mean()
            else:
                id_acc = torch.tensor(0.0, device=self.device)

        return total_loss, loc_acc.item(), id_acc.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_loc_acc = 0.0
        total_id_acc = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, loc_acc, id_acc = self._step(batch)
                total_loss += loss.item()
                total_loc_acc += loc_acc
                total_id_acc += id_acc
                steps += 1

        return total_loss / steps, total_loc_acc / steps, total_id_acc / steps

    def predict(self):
        print("Generating predictions for test set...")

        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()

        # Load Test Data
        test_dataset = InterleavedDataset("test", self.vocab, load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
        )

        predictions = []
        ids = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_ids = batch["id"]

                # Forward
                loc_logits, id_logits = self.model(input_ids, attention_mask)

                # Probabilities
                loc_probs = torch.sigmoid(loc_logits).squeeze(-1)  # (B, S)
                id_probs = torch.softmax(id_logits, dim=-1)  # (B, S, V)

                # Structural Masking:
                # Gaps only exist at odd indices (1, 3, 5...) in the interleaved sequence.
                # Mask even indices to 0 probability for localization.
                seq_len = input_ids.shape[1]
                gap_mask = torch.zeros((seq_len,), device=self.device)
                gap_mask[1::2] = 1.0

                loc_probs = loc_probs * gap_mask.unsqueeze(0)
                loc_probs = loc_probs * attention_mask

                # Score Fusion: Score(i, w) = P(Loc=i) * P(Word=w | Loc=i)
                # Shape: (B, S, V)
                scores = loc_probs.unsqueeze(-1) * id_probs

                # Find global maximum per sample
                batch_size = input_ids.shape[0]
                scores_flat = scores.view(batch_size, -1)
                best_flat_indices = torch.argmax(scores_flat, dim=1)

                # Decode indices
                best_gap_indices = best_flat_indices // Config.VOCAB_SIZE
                best_word_indices = best_flat_indices % Config.VOCAB_SIZE

                # Reconstruct Sentences
                input_ids_cpu = input_ids.cpu().numpy()
                best_gap_indices = best_gap_indices.cpu().numpy()
                best_word_indices = best_word_indices.cpu().numpy()

                for k in range(batch_size):
                    curr_input = input_ids_cpu[k]
                    curr_gap_idx = best_gap_indices[k]
                    curr_word_idx = best_word_indices[k]

                    # Get predicted word
                    pred_word = self.vocab.itos.get(curr_word_idx, Config.UNK_TOKEN)

                    # Extract original words from interleaved sequence
                    # Words are at even indices (0, 2, 4...)
                    words = []
                    for idx, token_id in enumerate(curr_input):
                        if token_id == 0:  # PAD
                            break
                        if idx % 2 == 0:
                            word = self.vocab.itos.get(token_id, Config.UNK_TOKEN)
                            words.append(word)

                    # Insert predicted word
                    # Gap at index `g` is between word `(g-1)/2` and `(g+1)/2`
                    # In a list of words, this corresponds to insertion index `(g+1)//2`
                    insert_idx = (curr_gap_idx + 1) // 2

                    # Clamp insertion index
                    if insert_idx > len(words):
                        insert_idx = len(words)

                    words.insert(insert_idx, pred_word)

                    sentence = " ".join(words)

                    ids.append(batch_ids[k])
                    predictions.append(sentence)

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        with open(Config.SUBMISSION_PATH, "w", encoding="utf-8") as f:
            f.write('id,"sentence"\n')
            for pid, sent in zip(ids, predictions):
                # Escape quotes according to spec: " -> ""
                sent_escaped = sent.replace('"', '""')
                f.write(f'{pid},"{sent_escaped}"\n')

        print("Submission generation complete.")
