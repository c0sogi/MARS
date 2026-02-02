import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import nltk
from tqdm.auto import tqdm
from typing import Optional, List, Dict, Tuple

from library.config import Config
from library.utils import get_logger
from library.vocab import Vocabulary

logger = get_logger("engine")


class Engine:
    """
    Handles training, evaluation, and inference for the Global-Localization Transformer.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        vocab: Vocabulary,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.device = device
        self.vocab = vocab
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_one_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        loc_loss_sum = 0.0
        id_loss_sum = 0.0
        align_loss_sum = 0.0
        count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            target_loc = batch["target_loc"].to(self.device)
            target_id = batch["target_id"].to(self.device)
            # gap_mask = batch["gap_mask"].to(self.device) # Not strictly used in loss, but useful for debug

            # Forward pass
            # Padding mask is automatically generated inside model if passed None,
            # but we can generate it here if needed. Model handles it.
            loc_logits, id_logits, hidden_states = self.model(input_ids)

            # Compute Loss
            loss_dict = self.criterion(
                loc_logits=loc_logits,
                id_logits=id_logits,
                hidden_states=hidden_states,
                target_loc=target_loc,
                target_id=target_id,
                embedding_layer=self.model.get_input_embeddings(),
            )

            loss = loss_dict["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Update metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            loc_loss_sum += loss_dict["loc_loss"].item() * batch_size
            id_loss_sum += loss_dict["id_loss"].item() * batch_size
            align_loss_sum += loss_dict["align_loss"].item() * batch_size
            count += batch_size

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / count if count > 0 else 0.0
        metrics = {
            "train_loss": avg_loss,
            "train_loc_loss": loc_loss_sum / count if count > 0 else 0.0,
            "train_id_loss": id_loss_sum / count if count > 0 else 0.0,
            "train_align_loss": align_loss_sum / count if count > 0 else 0.0,
        }

        logger.info(f"Epoch {epoch} Train Metrics: Loss={avg_loss:.6f}")
        return metrics

    def evaluate(self, dataloader, split: str = "val") -> Dict[str, float]:
        """
        Evaluates the model on the validation set.
        Computes Loss and Levenshtein Distance (on a subset).
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        # Levenshtein calculation vars
        lev_distances = []
        lev_sample_size = 1000  # Limit Levenshtein calc to first N samples to save time

        pbar = tqdm(dataloader, desc=f"Evaluating [{split}]", leave=False)

        with torch.no_grad():
            for i, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                target_loc = batch["target_loc"].to(self.device)
                target_id = batch["target_id"].to(self.device)
                gap_mask = batch["gap_mask"].to(self.device)

                loc_logits, id_logits, hidden_states = self.model(input_ids)

                loss_dict = self.criterion(
                    loc_logits=loc_logits,
                    id_logits=id_logits,
                    hidden_states=hidden_states,
                    target_loc=target_loc,
                    target_id=target_id,
                    embedding_layer=self.model.get_input_embeddings(),
                )

                batch_size = input_ids.size(0)
                total_loss += loss_dict["loss"].item() * batch_size
                count += batch_size

                # Calculate Levenshtein on a subset
                if len(lev_distances) < lev_sample_size:
                    self._compute_batch_levenshtein(
                        input_ids,
                        gap_mask,
                        loc_logits,
                        id_logits,
                        target_loc,
                        target_id,
                        lev_distances,
                    )

        avg_loss = total_loss / count if count > 0 else 0.0
        avg_lev = np.mean(lev_distances) if lev_distances else 0.0

        metrics = {f"{split}_loss": avg_loss, f"{split}_levenshtein": avg_lev}

        logger.info(
            f"[{split}] Loss: {avg_loss:.6f} | Levenshtein (subset): {avg_lev:.6f}"
        )
        return metrics

    def _compute_batch_levenshtein(
        self,
        input_ids,
        gap_mask,
        loc_logits,
        id_logits,
        target_loc,
        target_id,
        results_list,
    ):
        """
        Helper to compute Levenshtein distance for a batch.
        Reconstructs predicted and target sentences.
        """
        # 1. Prediction Strategy: Joint Probability
        # Mask invalid gaps
        loc_logits = loc_logits.masked_fill(gap_mask == 0, float("-inf"))

        log_p_loc = F.log_softmax(loc_logits, dim=1)  # (B, L)
        log_p_id = F.log_softmax(id_logits, dim=2)  # (B, L, V)

        # Joint Score: (B, L, V)
        # We want to maximize log(P(loc) * P(word|loc)) = log(P(loc)) + log(P(word|loc))
        joint_score = log_p_loc.unsqueeze(-1) + log_p_id

        # Flatten to find max
        b, l, v = joint_score.size()
        flat_score = joint_score.view(b, -1)
        best_flat_idx = flat_score.argmax(dim=1)

        pred_loc_idx = best_flat_idx // v
        pred_word_idx = best_flat_idx % v

        # 2. Reconstruction and Comparison
        input_ids_cpu = input_ids.cpu().tolist()
        target_loc_cpu = target_loc.cpu().tolist()
        target_id_cpu = target_id.cpu().tolist()
        pred_loc_cpu = pred_loc_idx.cpu().tolist()
        pred_word_cpu = pred_word_idx.cpu().tolist()

        for i in range(len(input_ids_cpu)):
            # Skip invalid targets (padding/truncation)
            if target_loc_cpu[i] == -100:
                continue

            # Extract original words (remove GAP and PAD)
            raw_tokens = [
                self.vocab.decode(tid)
                for tid in input_ids_cpu[i]
                if tid not in [self.vocab.pad_index, self.vocab.gap_index]
            ]

            # --- Construct Target Sentence ---
            # Target Gap Index in interleaved sequence is target_loc_cpu[i]
            # Insertion index in word list: (gap_idx + 1) // 2
            target_insert_idx = (target_loc_cpu[i] + 1) // 2
            target_word = self.vocab.decode(target_id_cpu[i])

            target_tokens = raw_tokens[:]
            # Safety check for insertion index
            if 0 <= target_insert_idx <= len(target_tokens):
                target_tokens.insert(target_insert_idx, target_word)

            target_sentence = " ".join(target_tokens)

            # --- Construct Predicted Sentence ---
            pred_insert_idx = (pred_loc_cpu[i] + 1) // 2
            pred_word = self.vocab.decode(pred_word_cpu[i])

            pred_tokens = raw_tokens[:]
            if 0 <= pred_insert_idx <= len(pred_tokens):
                pred_tokens.insert(pred_insert_idx, pred_word)

            pred_sentence = " ".join(pred_tokens)

            # Compute Distance
            dist = nltk.edit_distance(pred_sentence, target_sentence)
            results_list.append(dist)

    def predict_submission(self, dataloader, output_path: str):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.model.eval()
        results = []

        logger.info("Starting inference on test set...")
        pbar = tqdm(dataloader, desc="Inference", leave=True)

        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                gap_mask = batch["gap_mask"].to(self.device)
                row_ids = batch["row_id"]

                loc_logits, id_logits, _ = self.model(input_ids)

                # Mask invalid gaps
                loc_logits = loc_logits.masked_fill(gap_mask == 0, float("-inf"))

                # Joint Probability Maximization
                log_p_loc = F.log_softmax(loc_logits, dim=1)
                log_p_id = F.log_softmax(id_logits, dim=2)

                joint_score = log_p_loc.unsqueeze(-1) + log_p_id

                b, l, v = joint_score.size()
                best_flat_idx = joint_score.view(b, -1).argmax(dim=1)

                pred_loc_idx = best_flat_idx // v
                pred_word_idx = best_flat_idx % v

                # Reconstruction
                input_ids_cpu = input_ids.cpu().tolist()
                pred_loc_cpu = pred_loc_idx.cpu().tolist()
                pred_word_cpu = pred_word_idx.cpu().tolist()

                for i in range(b):
                    # Extract words
                    raw_tokens = [
                        self.vocab.decode(tid)
                        for tid in input_ids_cpu[i]
                        if tid not in [self.vocab.pad_index, self.vocab.gap_index]
                    ]

                    # Insert predicted word
                    insert_idx = (pred_loc_cpu[i] + 1) // 2
                    word_str = self.vocab.decode(pred_word_cpu[i])

                    if 0 <= insert_idx <= len(raw_tokens):
                        raw_tokens.insert(insert_idx, word_str)

                    sentence = " ".join(raw_tokens)
                    results.append({"id": row_ids[i], "sentence": sentence})

        # Save to CSV
        df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save with quoting to handle special characters/quotes in sentences
        import csv

        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Submission saved to {output_path}")
