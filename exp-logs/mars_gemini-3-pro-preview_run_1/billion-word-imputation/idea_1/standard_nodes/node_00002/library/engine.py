import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import MetricTracker, log_metrics, get_device


class Engine:
    """
    Engine class to handle training, validation, and submission generation.
    """

    def __init__(self, model, optimizer, scheduler, vocab, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocab = vocab
        self.device = device if device else get_device()

        # Initialize Loss Function with Class Balancing
        # The [NO_INSERT] class is the vast majority. We down-weight it.
        self.weight = torch.ones(len(vocab), device=self.device)
        if vocab.no_insert_token_id < len(vocab):
            self.weight[vocab.no_insert_token_id] = Config.NO_INSERT_WEIGHT

        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=-100)

    def train_one_epoch(self, dataloader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        tracker = MetricTracker()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            logits = self.model(input_ids)  # (Batch, Seq_Len, Vocab)

            # Flatten for loss calculation
            loss = self.criterion(logits.view(-1, len(self.vocab)), targets.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            # Step scheduler (assuming batch-level scheduler like OneCycleLR)
            if self.scheduler is not None:
                self.scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                mask = targets != -100
                correct = (preds == targets) & mask
                total = mask.sum()
                accuracy = (
                    correct.sum() / total
                    if total > 0
                    else torch.tensor(0.0, device=self.device)
                )

                tracker.update(
                    {"loss": loss.item(), "accuracy": accuracy.item()},
                    n=input_ids.size(0),
                )

        metrics = tracker.get_averages()
        log_metrics(metrics, prefix=f"Train Epoch {epoch}")
        return metrics

    def validate(self, dataloader, epoch=None):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, len(self.vocab)), targets.view(-1)
                )

                preds = torch.argmax(logits, dim=-1)
                mask = targets != -100
                correct = (preds == targets) & mask
                total = mask.sum()
                accuracy = (
                    correct.sum() / total
                    if total > 0
                    else torch.tensor(0.0, device=self.device)
                )

                tracker.update(
                    {"loss": loss.item(), "accuracy": accuracy.item()},
                    n=input_ids.size(0),
                )

        metrics = tracker.get_averages()
        prefix = f"Val Epoch {epoch}" if epoch is not None else "Val"
        log_metrics(metrics, prefix=prefix)
        return metrics

    def generate_submission(self, dataloader, output_file):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        results = []

        # Identify special tokens to mask
        pad_id = self.vocab.stoi[self.vocab.TOKEN_PAD]
        unk_id = self.vocab.stoi[self.vocab.TOKEN_UNK]
        no_insert_id = self.vocab.stoi[self.vocab.TOKEN_NO_INSERT]
        start_id = self.vocab.stoi[self.vocab.TOKEN_START]
        end_id = self.vocab.stoi[self.vocab.TOKEN_END]

        mask_ids = [pad_id, unk_id, no_insert_id, start_id, end_id]

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                ids = batch["id"].cpu().numpy()

                logits = self.model(input_ids)  # (B, S, V)
                probs = torch.softmax(logits, dim=-1)

                # Mask special tokens in the vocabulary dimension
                for mid in mask_ids:
                    probs[:, :, mid] = 0.0

                batch_probs = probs.cpu().numpy()
                batch_input = input_ids.cpu().numpy()

                for i in range(len(ids)):
                    p = batch_probs[i]  # (S, V)
                    seq = batch_input[i]  # (S,)

                    # Mask invalid gap positions
                    # 1. Gap 0: After [START] (Before first word). Mask it.
                    p[0, :] = 0.0

                    # 2. Find [END] token to determine sequence boundary
                    end_indices = np.where(seq == end_id)[0]
                    if len(end_indices) > 0:
                        end_idx = end_indices[0]
                        # Mask everything from [END] onwards
                        p[end_idx:, :] = 0.0
                        # Mask the gap before [END] (After last word).
                        if end_idx > 0:
                            p[end_idx - 1, :] = 0.0
                    else:
                        # Fallback if no END token found (should not happen)
                        end_idx = len(seq)

                    # Find the position and word with maximum probability
                    flat_idx = np.argmax(p)
                    gap_idx, word_idx = np.unravel_index(flat_idx, p.shape)

                    # Reconstruct the sentence
                    # Extract original words (excluding START and END)
                    words = [self.vocab.get_token(tid) for tid in seq[1:end_idx]]
                    pred_word = self.vocab.get_token(word_idx)

                    # Insert predicted word
                    # gap_idx corresponds to the index in `seq` (input_ids).
                    # We predicted the gap AFTER `seq[gap_idx]`.
                    # `words` list starts from `seq[1]`.
                    # So `seq[gap_idx]` corresponds to `words[gap_idx - 1]`.
                    # We want to insert AFTER `words[gap_idx - 1]`.
                    # The insertion index is `(gap_idx - 1) + 1` = `gap_idx`.

                    # Safety check for index bounds
                    insert_pos = min(gap_idx, len(words))
                    words.insert(insert_pos, pred_word)

                    sentence = " ".join(words)
                    results.append((ids[i], sentence))

        # Save to CSV using pandas
        # quoting=csv.QUOTE_NONNUMERIC ensures all non-numeric fields (sentences) are quoted.
        df_sub = pd.DataFrame(results, columns=["id", "sentence"])
        df_sub.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return df_sub
