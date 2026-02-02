import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.models import SiameseDANRanker, ShallowCNNReader
from library.data_loader import get_dataloaders


class EarlyStopper:
    def __init__(
        self, patience=Config.EARLY_STOPPING_PATIENCE, delta=Config.EARLY_STOPPING_DELTA
    ):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    def __init__(self, debug_sample_size=None):
        self.device = Config.get_device()
        print(f"Trainer initialized on device: {self.device}")

        # Data
        self.loaders, self.vocab_encoder = get_dataloaders(debug_sample_size)
        vocab_size = len(self.vocab_encoder)
        pad_idx = self.vocab_encoder.pad_idx

        # Models
        self.ranker = SiameseDANRanker(vocab_size, padding_idx=pad_idx).to(self.device)
        self.reader = ShallowCNNReader(vocab_size, padding_idx=pad_idx).to(self.device)

        # Optimizers
        self.ranker_optimizer = optim.Adam(
            self.ranker.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.reader_optimizer = optim.Adam(
            self.reader.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Losses
        self.ranker_criterion = nn.MarginRankingLoss(margin=Config.RANKING_MARGIN)
        self.reader_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def train_ranker_epoch(self, epoch_idx):
        self.ranker.train()
        total_loss = 0.0

        for batch_idx, (q_ids, pos_ids, neg_ids) in enumerate(
            self.loaders["ranker_train"]
        ):
            q_ids = q_ids.to(self.device)
            pos_ids = pos_ids.to(self.device)
            neg_ids = neg_ids.to(self.device)  # Shape: (B, Num_Neg, L)

            self.ranker_optimizer.zero_grad()

            # Forward
            pos_scores = self.ranker(q_ids, pos_ids)  # (B,)
            neg_scores = self.ranker(q_ids, neg_ids)  # (B, Num_Neg)

            # Prepare for MarginRankingLoss: inputs x1, x2, y
            # We want pos_score > neg_score, so y=1
            # Expand pos_scores to match neg_scores shape
            pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)

            # Flatten for loss computation
            target = torch.ones_like(neg_scores).to(self.device)

            loss = self.ranker_criterion(
                pos_scores_expanded.flatten(), neg_scores.flatten(), target.flatten()
            )

            loss.backward()
            self.ranker_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.loaders["ranker_train"])
        print(f"Ranker Epoch {epoch_idx+1} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate_ranker(self):
        self.ranker.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for q_ids, pos_ids, neg_ids in self.loaders["ranker_val"]:
                q_ids = q_ids.to(self.device)
                pos_ids = pos_ids.to(self.device)
                neg_ids = neg_ids.to(self.device)

                pos_scores = self.ranker(q_ids, pos_ids)
                neg_scores = self.ranker(q_ids, neg_ids)

                pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
                target = torch.ones_like(neg_scores).to(self.device)

                loss = self.ranker_criterion(
                    pos_scores_expanded.flatten(),
                    neg_scores.flatten(),
                    target.flatten(),
                )
                total_loss += loss.item()

                # Accuracy: Pos score > Max(Neg scores)
                max_neg_scores, _ = torch.max(neg_scores, dim=1)
                correct += (pos_scores > max_neg_scores).sum().item()
                total_samples += q_ids.size(0)

        avg_loss = total_loss / len(self.loaders["ranker_val"])
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        print(f"Ranker Validation | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.6f}")
        return avg_loss

    def train_reader_epoch(self, epoch_idx):
        self.reader.train()
        total_loss = 0.0

        for input_ids, start_targets, end_targets in self.loaders["reader_train"]:
            input_ids = input_ids.to(self.device)
            start_targets = start_targets.to(self.device)
            end_targets = end_targets.to(self.device)

            self.reader_optimizer.zero_grad()

            start_logits, end_logits = self.reader(input_ids)

            loss = self.reader_criterion(
                start_logits, start_targets
            ) + self.reader_criterion(end_logits, end_targets)

            loss.backward()
            self.reader_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.loaders["reader_train"])
        print(f"Reader Epoch {epoch_idx+1} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate_reader(self):
        self.reader.eval()
        total_loss = 0.0
        exact_match = 0
        total_samples = 0

        with torch.no_grad():
            for input_ids, start_targets, end_targets in self.loaders["reader_val"]:
                input_ids = input_ids.to(self.device)
                start_targets = start_targets.to(self.device)
                end_targets = end_targets.to(self.device)

                start_logits, end_logits = self.reader(input_ids)

                loss = self.reader_criterion(
                    start_logits, start_targets
                ) + self.reader_criterion(end_logits, end_targets)
                total_loss += loss.item()

                # Predictions
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)

                exact_match += (
                    ((start_preds == start_targets) & (end_preds == end_targets))
                    .sum()
                    .item()
                )
                total_samples += input_ids.size(0)

        avg_loss = total_loss / len(self.loaders["reader_val"])
        em_score = exact_match / total_samples if total_samples > 0 else 0.0
        print(f"Reader Validation | Loss: {avg_loss:.6f} | Exact Match: {em_score:.6f}")
        return avg_loss

    def run_training(self):
        # 1. Train Ranker
        print("\n--- Starting Ranker Training ---")
        stopper = EarlyStopper()
        ranker_path = os.path.join(Config.CACHE_DIR, "ranker_best.pth")

        for epoch in range(Config.EPOCHS):
            self.train_ranker_epoch(epoch)
            val_loss = self.validate_ranker()

            if stopper.best_loss == val_loss:
                torch.save(self.ranker.state_dict(), ranker_path)

            if stopper(val_loss):
                print(f"Early stopping Ranker at epoch {epoch+1}")
                break

        # Load best ranker
        if os.path.exists(ranker_path):
            self.ranker.load_state_dict(
                torch.load(ranker_path, map_location=self.device)
            )

        # 2. Train Reader
        print("\n--- Starting Reader Training ---")
        stopper = EarlyStopper()
        reader_path = os.path.join(Config.CACHE_DIR, "reader_best.pth")

        for epoch in range(Config.EPOCHS):
            self.train_reader_epoch(epoch)
            val_loss = self.validate_reader()

            if stopper.best_loss == val_loss:
                torch.save(self.reader.state_dict(), reader_path)

            if stopper(val_loss):
                print(f"Early stopping Reader at epoch {epoch+1}")
                break

        if os.path.exists(reader_path):
            self.reader.load_state_dict(
                torch.load(reader_path, map_location=self.device)
            )

    def generate_submission(self):
        print("\n--- Generating Submission ---")
        self.ranker.eval()
        self.reader.eval()

        results = []

        # Determine UNK/SEP token for concatenation
        sep_token_id = self.vocab_encoder.unk_idx

        with torch.no_grad():
            for example_id, q_ids, cand_tensors, cand_meta in self.loaders["test"]:
                # Unpack batch size 1
                q_ids = q_ids.to(self.device).unsqueeze(0)  # (1, Q_Len)
                cand_tensors = cand_tensors.to(self.device).unsqueeze(
                    0
                )  # (1, Num_Cand, Ctx_Len)

                # 1. Rank Candidates
                # cand_tensors shape: (1, K, L)
                scores = self.ranker(q_ids, cand_tensors)  # (1, K)
                scores = scores.squeeze(0)  # (K,)

                if scores.numel() == 0:
                    # No candidates found
                    results.append(f"{example_id}_long,")
                    results.append(f"{example_id}_short,")
                    continue

                best_score, best_idx = torch.max(scores, dim=0)
                best_idx = best_idx.item()
                best_score = best_score.item()

                # Confidence Threshold
                if best_score < Config.CONFIDENCE_THRESHOLD:
                    results.append(f"{example_id}_long,")
                    results.append(f"{example_id}_short,")
                    continue

                # 2. Prepare Long Answer Prediction
                best_cand_meta = cand_meta[best_idx]
                long_ans_str = f"{best_cand_meta['start_token_idx']}:{best_cand_meta['end_token_idx']}"
                results.append(f"{example_id}_long,{long_ans_str}")

                # 3. Reader Extraction
                # Construct input: Q + SEP + Best_Candidate
                # Note: q_ids and cand_tensors are padded. We need actual tokens to concatenate cleanly
                # or just concatenate tensors and ignore padding in middle (less ideal).
                # Better: reconstruct from original indices or careful slicing.

                # Get raw indices
                q_raw = q_ids[0]
                ctx_raw = cand_tensors[0, best_idx]

                # Remove padding
                q_valid = q_raw[q_raw != self.vocab_encoder.pad_idx]
                ctx_valid = ctx_raw[ctx_raw != self.vocab_encoder.pad_idx]

                # Concatenate: Q + SEP + CTX
                sep_tensor = torch.tensor([sep_token_id], device=self.device)
                reader_input = torch.cat([q_valid, sep_tensor, ctx_valid]).unsqueeze(
                    0
                )  # (1, Seq_Len)

                # Clip to max length if needed (though model handles it via embedding,
                # but we need to match training max len logic if strict)
                # Training max len was MAX_Q + MAX_CTX. Here we just feed it.

                start_logits, end_logits = self.reader(reader_input)

                start_pred = torch.argmax(start_logits, dim=1).item()
                end_pred = torch.argmax(end_logits, dim=1).item()

                # 4. Map back to global coordinates
                # Offset caused by Q + SEP
                offset = len(q_valid) + 1

                local_start = start_pred - offset
                local_end = end_pred - offset

                # Validate span
                if (
                    local_start >= 0
                    and local_end >= local_start
                    and local_end < len(ctx_valid)
                ):
                    global_start = best_cand_meta["start_token_idx"] + local_start
                    global_end = best_cand_meta["start_token_idx"] + local_end
                    # NQ format is usually inclusive start, exclusive end?
                    # The task description says "start:end token indices".
                    # Usually end is exclusive in Python slicing, but inclusive in some metrics.
                    # Standard NQ evaluation treats end as exclusive.
                    # Our training logic used `end_token_idx` from split which was exclusive.
                    # So global_end is exclusive.
                    short_ans_str = f"{global_start}:{global_end}"
                    results.append(f"{example_id}_short,{short_ans_str}")
                else:
                    # Invalid span (e.g. start < offset means answer in question part)
                    results.append(f"{example_id}_short,")

        # Write submission
        with open(Config.SUBMISSION_PATH, "w") as f:
            f.write("example_id,PredictionString\n")
            f.write("\n".join(results))

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
