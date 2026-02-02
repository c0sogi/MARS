import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from library.config import Config
from library.data import YES_NO_MAP

# Invert YES_NO_MAP for inference
IDX_TO_YES_NO = {v: k for k, v in YES_NO_MAP.items()}


class Engine:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer = None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()  # For Yes/No
        self.span_loss = nn.CrossEntropyLoss(
            ignore_index=0
        )  # Index 0 is PAD, usually treated as no-answer/null in span tasks

    def calculate_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the weighted multi-task loss.
        """
        # 1. Ranking Loss (Binary Classification)
        ranking_loss = self.bce_loss(
            outputs["ranking_logits"], targets["ranking_labels"]
        )

        # 2. Span Loss (Start and End indices)
        start_loss = self.span_loss(outputs["start_logits"], targets["start_labels"])
        end_loss = self.span_loss(outputs["end_logits"], targets["end_labels"])

        # 3. Yes/No Loss (Multi-class Classification)
        yes_no_loss = self.ce_loss(outputs["yes_no_logits"], targets["yes_no_labels"])

        # Weighted Sum
        total_loss = (
            Config.LOSS_WEIGHT_RANKING * ranking_loss
            + Config.LOSS_WEIGHT_SPAN * (start_loss + end_loss) / 2.0
            + Config.LOSS_WEIGHT_YESNO * yes_no_loss
        )

        return total_loss

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(self.device)

            # Move targets to device
            targets = {
                "ranking_labels": batch["ranking_labels"].to(self.device),
                "start_labels": batch["start_labels"].to(self.device),
                "end_labels": batch["end_labels"].to(self.device),
                "yes_no_labels": batch["yes_no_labels"].to(self.device),
            }

            self.optimizer.zero_grad()

            outputs = self.model(input_ids)
            loss = self.calculate_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            count += input_ids.size(0)

        return total_loss / count if count > 0 else 0.0

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0

        # Metrics trackers
        correct_rank = 0
        correct_yes_no = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)

                targets = {
                    "ranking_labels": batch["ranking_labels"].to(self.device),
                    "start_labels": batch["start_labels"].to(self.device),
                    "end_labels": batch["end_labels"].to(self.device),
                    "yes_no_labels": batch["yes_no_labels"].to(self.device),
                }

                outputs = self.model(input_ids)
                loss = self.calculate_loss(outputs, targets)

                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                count += batch_size

                # Simple accuracy metrics
                # Ranking: sigmoid > 0.5 vs label
                rank_preds = (torch.sigmoid(outputs["ranking_logits"]) > 0.5).float()
                correct_rank += (rank_preds == targets["ranking_labels"]).sum().item()

                # Yes/No: argmax vs label
                yn_preds = torch.argmax(outputs["yes_no_logits"], dim=1)
                correct_yes_no += (yn_preds == targets["yes_no_labels"]).sum().item()

        avg_loss = total_loss / count if count > 0 else 0.0
        acc_rank = correct_rank / count if count > 0 else 0.0
        acc_yn = correct_yes_no / count if count > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "val_rank_acc": acc_rank,
            "val_yes_no_acc": acc_yn,
        }

    def predict(self, dataloader: torch.utils.data.DataLoader, test_df: pd.DataFrame):
        """
        Runs inference on the test set and generates the submission file.
        """
        self.model.eval()

        # Store predictions grouped by example_id
        # example_id -> list of candidate predictions
        all_predictions = {}

        print("Running inference...")

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)

                outputs = self.model(input_ids)

                # Move to CPU
                ranking_scores = torch.sigmoid(outputs["ranking_logits"]).cpu().numpy()
                start_logits = outputs["start_logits"].cpu().numpy()
                end_logits = outputs["end_logits"].cpu().numpy()
                yes_no_logits = outputs["yes_no_logits"].cpu().numpy()

                # Metadata
                example_ids = batch["example_ids"]
                cand_indices = batch["candidate_indices"]

                # Input sequences (to determine question length for offset calculation)
                input_seqs = input_ids.cpu().numpy()

                for i in range(len(example_ids)):
                    eid = example_ids[i]
                    c_idx = cand_indices[i]
                    rank_score = ranking_scores[i]

                    # Determine question length (find SEP token)
                    # vocab[SEP] is required. We can infer it from input_seqs if we know SEP ID.
                    # Or we can assume the structure [Q] [SEP] [Cand].
                    # We need the vocab to find SEP ID.
                    sep_id = self.model.vocab[Config.SEP_TOKEN]

                    try:
                        # np.where returns tuple of arrays
                        sep_pos = np.where(input_seqs[i] == sep_id)[0][0]
                        # Candidate starts after SEP
                        cand_offset_in_seq = sep_pos + 1
                    except IndexError:
                        # Fallback if SEP not found (should not happen with correct collator)
                        cand_offset_in_seq = 0

                    # Span Prediction
                    s_idx = np.argmax(start_logits[i])
                    e_idx = np.argmax(end_logits[i])

                    # Yes/No Prediction
                    yn_idx = np.argmax(yes_no_logits[i])

                    if eid not in all_predictions:
                        all_predictions[eid] = []

                    all_predictions[eid].append(
                        {
                            "cand_idx": c_idx,
                            "rank_score": rank_score,
                            "start_seq_idx": s_idx,
                            "end_seq_idx": e_idx,
                            "cand_offset_in_seq": cand_offset_in_seq,
                            "yes_no_idx": yn_idx,
                        }
                    )

        # Process predictions to generate submission rows
        submission_rows = []

        # We need to look up actual candidate offsets from test_df
        # Create a quick lookup for test_df candidates
        # test_df has columns: example_id, candidates (list of tuples)
        # We can index by example_id
        test_data_map = test_df.set_index("example_id")["candidates"].to_dict()

        print("Formatting submission...")

        # Ensure we cover all IDs in test_df
        all_test_ids = test_df["example_id"].unique()

        for eid in all_test_ids:
            # Default predictions
            long_pred_str = ""
            short_pred_str = ""

            if eid in all_predictions:
                preds = all_predictions[eid]

                # 1. Select best long answer candidate
                # Sort by ranking score descending
                preds.sort(key=lambda x: x["rank_score"], reverse=True)
                best_pred = preds[0]

                if best_pred["rank_score"] >= Config.LONG_ANSWER_THRESHOLD:
                    # Get candidate info from raw data
                    candidates = test_data_map.get(str(eid), [])
                    if best_pred["cand_idx"] < len(candidates):
                        c_start, c_end, _ = candidates[best_pred["cand_idx"]]

                        # Set Long Answer Prediction
                        long_pred_str = f"{c_start}:{c_end}"

                        # 2. Determine Short Answer
                        # Logic: If we have a valid long answer, we check for a short answer within it.
                        # The model predicted start/end indices relative to the input sequence.
                        # We need to map them to global document tokens.

                        s_seq = best_pred["start_seq_idx"]
                        e_seq = best_pred["end_seq_idx"]
                        offset = best_pred["cand_offset_in_seq"]

                        # Check validity:
                        # 1. Indices must be within the candidate portion of the sequence
                        # 2. Start <= End
                        if s_seq >= offset and e_seq >= offset and s_seq <= e_seq:
                            # Map to global
                            # relative index in candidate = s_seq - offset
                            # global index = c_start + relative

                            global_s = c_start + (s_seq - offset)
                            global_e = c_start + (e_seq - offset)

                            # Ensure global end doesn't exceed candidate end
                            if global_e < c_end:
                                # Check Yes/No first (priority over span if YES/NO is predicted?)
                                # NQ rules: Short answer can be YES/NO OR a span.
                                # Usually YES/NO is mutually exclusive with span in annotations,
                                # but here we have a separate head.
                                # Logic: If Yes/No is YES or NO, output that. Else output span.

                                yn_label = IDX_TO_YES_NO.get(
                                    best_pred["yes_no_idx"], "NONE"
                                )

                                if yn_label in ["YES", "NO"]:
                                    short_pred_str = yn_label
                                else:
                                    # Output span (inclusive end index for NQ submission usually?
                                    # NQ format is start:end (exclusive) usually, but sample says token indices.
                                    # The task description says "start:end token indices".
                                    # Usually NQ is start:end byte, but here "simplified" uses tokens.
                                    # Let's assume standard python slice notation start:end (exclusive)
                                    # or inclusive?
                                    # Looking at sample: -545833482873225036_long,105:200
                                    # Let's output global_s:global_e+1 to be safe for token span
                                    short_pred_str = f"{global_s}:{global_e + 1}"

            # Append rows
            submission_rows.append(
                {"example_id": f"{eid}_long", "PredictionString": long_pred_str}
            )
            submission_rows.append(
                {"example_id": f"{eid}_short", "PredictionString": short_pred_str}
            )

        # Save submission
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
