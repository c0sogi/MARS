import os
import json
import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from library.config import config
from library.data_utils import build_vocab
from library.dataset import NQDataset
from library.model import KernelPoolingNetwork
from library.loss import MultiTaskLoss


class Trainer:
    def __init__(self, load_cached_data=True, limit_size=None):
        """
        Trainer for the Kernel-Pooling Interaction Network.

        Args:
            load_cached_data (bool): Whether to load pre-processed data from cache.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.set_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Prepare Data Resources
        self.vocab = build_vocab(load_cached_data=load_cached_data)

        # 2. Initialize Model
        self.model = KernelPoolingNetwork(self.vocab).to(self.device)

        # 3. Setup Optimization
        self.criterion = MultiTaskLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Data Loading parameters
        self.load_cached_data = load_cached_data
        self.limit_size = limit_size

    def set_seeds(self):
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        # Prepare Datasets
        train_dataset = NQDataset(
            split="train",
            vocab=self.vocab,
            load_cached_data=self.load_cached_data,
            limit_size=self.limit_size,
        )
        val_dataset = NQDataset(
            split="val",
            vocab=self.vocab,
            load_cached_data=self.load_cached_data,
            limit_size=self.limit_size,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting training...")
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

            # Train Step
            train_metrics = self.train_epoch(train_loader)
            print(
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Rank Acc: {train_metrics['acc_rank']:.6f} | "
                f"Span Acc: {train_metrics['acc_span']:.6f} | "
                f"Y/N Acc: {train_metrics['acc_yesno']:.6f}"
            )

            # Validation Step
            val_metrics = self.validate(val_loader)
            print(
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Rank Acc: {val_metrics['acc_rank']:.6f} | "
                f"Span Acc: {val_metrics['acc_span']:.6f} | "
                f"Y/N Acc: {val_metrics['acc_yesno']:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(self.model.state_dict(), config.MODEL_CHECKPOINT_PATH)
                print(f"New best model saved to {config.MODEL_CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                print(
                    f"Early stopping counter: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
                )
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        # Accuracy trackers
        correct_rank = 0
        correct_span = 0
        correct_yesno = 0
        total_samples = 0

        for batch in loader:
            # Move batch to device
            question = batch["question"].to(self.device)
            candidate = batch["candidate"].to(self.device)

            targets = {
                "label_long": batch["label_long"].to(self.device),
                "label_span_start": batch["label_span_start"].to(self.device),
                "label_span_end": batch["label_span_end"].to(self.device),
                "label_yesno": batch["label_yesno"].to(self.device),
            }

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(question, candidate)

            # Loss
            loss, _ = self.criterion(outputs, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Metrics
            batch_size = question.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Ranking Accuracy (Binary: > 0.5 is positive)
            # outputs['long_score'] is logits
            preds_long = (
                torch.sigmoid(outputs["long_score"].squeeze(-1)) > 0.5
            ).float()
            correct_rank += (preds_long == targets["label_long"]).sum().item()

            # Span Accuracy (Exact Match of start AND end)
            pred_start = torch.argmax(outputs["start_logits"], dim=1)
            pred_end = torch.argmax(outputs["end_logits"], dim=1)
            match_start = pred_start == targets["label_span_start"]
            match_end = pred_end == targets["label_span_end"]
            correct_span += (match_start & match_end).sum().item()

            # Yes/No Accuracy
            pred_yesno = torch.argmax(outputs["yesno_logits"], dim=1)
            correct_yesno += (pred_yesno == targets["label_yesno"]).sum().item()

        return {
            "loss": total_loss / total_samples,
            "acc_rank": correct_rank / total_samples,
            "acc_span": correct_span / total_samples,
            "acc_yesno": correct_yesno / total_samples,
        }

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        correct_rank = 0
        correct_span = 0
        correct_yesno = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                question = batch["question"].to(self.device)
                candidate = batch["candidate"].to(self.device)

                targets = {
                    "label_long": batch["label_long"].to(self.device),
                    "label_span_start": batch["label_span_start"].to(self.device),
                    "label_span_end": batch["label_span_end"].to(self.device),
                    "label_yesno": batch["label_yesno"].to(self.device),
                }

                outputs = self.model(question, candidate)
                loss, _ = self.criterion(outputs, targets)

                batch_size = question.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Metrics
                preds_long = (
                    torch.sigmoid(outputs["long_score"].squeeze(-1)) > 0.5
                ).float()
                correct_rank += (preds_long == targets["label_long"]).sum().item()

                pred_start = torch.argmax(outputs["start_logits"], dim=1)
                pred_end = torch.argmax(outputs["end_logits"], dim=1)
                match_start = pred_start == targets["label_span_start"]
                match_end = pred_end == targets["label_span_end"]
                correct_span += (match_start & match_end).sum().item()

                pred_yesno = torch.argmax(outputs["yesno_logits"], dim=1)
                correct_yesno += (pred_yesno == targets["label_yesno"]).sum().item()

        return {
            "loss": total_loss / total_samples,
            "acc_rank": correct_rank / total_samples,
            "acc_span": correct_span / total_samples,
            "acc_yesno": correct_yesno / total_samples,
        }

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves submission.csv.
        Requires reading raw test file to map candidate indices to token offsets.
        """
        # Load best model
        if os.path.exists(config.MODEL_CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(config.MODEL_CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded best model from {config.MODEL_CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()

        # Prepare Test Data
        test_dataset = NQDataset(
            split="test",
            vocab=self.vocab,
            load_cached_data=self.load_cached_data,
            limit_size=self.limit_size,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        print("Running inference on test set...")

        # Store predictions: example_id -> list of candidate predictions
        # Each prediction: {cand_idx, long_score, start_idx, end_idx, yesno_idx}
        all_predictions = {}

        with torch.no_grad():
            for batch in test_loader:
                question = batch["question"].to(self.device)
                candidate = batch["candidate"].to(self.device)

                outputs = self.model(question, candidate)

                # Extract scores
                long_scores = (
                    torch.sigmoid(outputs["long_score"]).cpu().numpy().flatten()
                )
                start_idxs = torch.argmax(outputs["start_logits"], dim=1).cpu().numpy()
                end_idxs = torch.argmax(outputs["end_logits"], dim=1).cpu().numpy()
                yesno_idxs = torch.argmax(outputs["yesno_logits"], dim=1).cpu().numpy()

                example_ids = batch["example_id"]  # List of strings
                cand_idxs = batch["candidate_index"].numpy()

                for i in range(len(example_ids)):
                    eid = example_ids[i]
                    if eid not in all_predictions:
                        all_predictions[eid] = []

                    all_predictions[eid].append(
                        {
                            "cand_idx": cand_idxs[i],
                            "long_score": long_scores[i],
                            "start_rel": start_idxs[i],
                            "end_rel": end_idxs[i],
                            "yesno_idx": yesno_idxs[i],
                        }
                    )

        # Process predictions and generate submission file
        print("Generating submission file...")

        # We need to read the raw test file to get global offsets
        submission_rows = []
        yn_labels = {0: "", 1: "YES", 2: "NO"}  # 0 is NONE -> Blank

        with open(config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ex_id = str(entry["example_id"])

                if ex_id not in all_predictions:
                    # Should not happen if dataset is aligned, but handle gracefully
                    submission_rows.append([f"{ex_id}_long", ""])
                    submission_rows.append([f"{ex_id}_short", ""])
                    continue

                preds = all_predictions[ex_id]

                # Find best candidate by long_score
                best_pred = max(preds, key=lambda x: x["long_score"])

                # Thresholding
                if best_pred["long_score"] < config.LONG_ANSWER_THRESHOLD:
                    # No answer
                    submission_rows.append([f"{ex_id}_long", ""])
                    submission_rows.append([f"{ex_id}_short", ""])
                else:
                    # Get candidate info from raw entry
                    candidates = entry["long_answer_candidates"]
                    c_idx = best_pred["cand_idx"]

                    if c_idx < len(candidates):
                        cand_info = candidates[c_idx]
                        global_start = cand_info["start_token"]
                        global_end = cand_info["end_token"]

                        # Long Answer String
                        long_str = f"{global_start}:{global_end}"

                        # Short Answer String
                        # Start/End relative to candidate
                        s_rel = best_pred["start_rel"]
                        e_rel = best_pred["end_rel"]

                        # Map to global
                        s_global = global_start + s_rel
                        e_global = global_start + e_rel

                        # Validate span
                        if (
                            s_global < global_end
                            and e_global < global_end
                            and s_global <= e_global
                        ):
                            short_str = f"{s_global}:{e_global + 1}"  # +1 because range is exclusive in python/standard?
                            # NQ format usually expects token indices.
                            # If the format is start:end (inclusive:exclusive), then +1.
                            # Standard NQ evaluation uses inclusive start, exclusive end.
                        else:
                            short_str = ""

                        # Yes/No
                        yn_idx = best_pred["yesno_idx"]
                        yn_str = yn_labels.get(yn_idx, "")

                        # If Yes/No is present, it overrides span for short answer text in some contexts,
                        # but the submission format usually asks for span OR yes/no.
                        # The instructions say: "a set of start:end token indices, b) a YES/NO answer... c) BLANK"
                        # Usually for NQ, if YES/NO, the span is still relevant or blank?
                        # The prompt example shows: "-785..._short,YES".
                        # So if YES/NO is predicted, we output that. Else span.

                        final_short_str = yn_str if yn_str else short_str

                        submission_rows.append([f"{ex_id}_long", long_str])
                        submission_rows.append([f"{ex_id}_short", final_short_str])
                    else:
                        # Fallback
                        submission_rows.append([f"{ex_id}_long", ""])
                        submission_rows.append([f"{ex_id}_short", ""])

        # Write CSV
        sub_df = pd.DataFrame(
            submission_rows, columns=["example_id", "PredictionString"]
        )
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
