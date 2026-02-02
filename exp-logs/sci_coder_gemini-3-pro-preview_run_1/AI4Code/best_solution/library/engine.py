import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import (
    FEATURE_DIR,
    VAL_METADATA_PATH,
    SUBMISSION_PATH,
    MAX_ANCHOR_SEQ_LEN,
)
from library.utils import compute_kendall_tau


class Engine:
    def __init__(self, model, device, optimizer=None):
        """
        Engine for training, validating, and inferencing the CAAN model.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer

        # Caches for reconstruction maps to avoid reloading parquet files repeatedly
        self.val_map = None
        self.val_gt = None
        self.test_map = None

    def _load_metadata_map(self, split):
        """
        Loads the cell ID and type information from the cached parquet file.
        Returns a dict: nb_id -> (sorted_code_ids, ordered_md_ids)
        """
        parquet_path = os.path.join(FEATURE_DIR, f"{split}_features.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Features file not found: {parquet_path}")

        # Load only necessary columns to save memory
        df = pd.read_parquet(
            parquet_path, columns=["id", "cell_id", "cell_type", "rank_in_code"]
        )

        meta_map = {}
        # Group by notebook ID. sort=False preserves occurrence order (crucial for MD cells)
        for nb_id, group in df.groupby("id", sort=False):
            code_mask = group["cell_type"] == "code"
            md_mask = group["cell_type"] == "markdown"

            # Code cells must be sorted by their rank to match the model's anchor sequence
            code_ids = group[code_mask].sort_values("rank_in_code")["cell_id"].tolist()

            # Markdown cells are kept in the order they appear in the dataframe
            # (which matches the order in NotebookDataset/Preprocessor)
            md_ids = group[md_mask]["cell_id"].tolist()

            meta_map[nb_id] = (code_ids, md_ids)

        return meta_map

    def train_one_epoch(self, dataloader, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        # Use tqdm for progress tracking
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train", mininterval=30)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for batch in pbar:
            # Move data to device
            code_emb = batch["code_emb"].to(self.device)
            code_mask = batch["code_mask"].to(self.device)
            md_emb = batch["md_emb"].to(self.device)
            md_mask = batch["md_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            # Logits shape: (Batch, Max_MD, Max_Code + 1)
            logits = self.model(code_emb, code_mask, md_emb, md_mask)

            # Flatten for CrossEntropyLoss
            # Logits: (B * L_md, Classes)
            # Labels: (B * L_md)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = loss_fn(logits_flat, labels_flat)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def validate(self, dataloader, dataset):
        """
        Evaluates the model on the validation set.
        Computes CrossEntropyLoss and Kendall Tau correlation.
        """
        # Load metadata if not already loaded
        if self.val_map is None:
            print("Loading validation metadata for reconstruction...")
            self.val_map = self._load_metadata_map("val")

            # Load ground truth orders
            df_val = pd.read_csv(VAL_METADATA_PATH)
            self.val_gt = dict(zip(df_val["id"], df_val["cell_order"]))

        self.model.eval()
        total_loss = 0.0
        predictions = []
        ground_truths = []

        # Track dataset index to retrieve notebook IDs
        current_idx = 0

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", mininterval=30):
                code_emb = batch["code_emb"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_emb = batch["md_emb"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward
                logits = self.model(code_emb, code_mask, md_emb, md_mask)

                # Compute Loss
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = loss_fn(logits_flat, labels_flat)
                total_loss += loss.item()

                # --- Reconstruction for Kendall Tau ---

                # 1. Calculate Expected Rank
                # Probabilities: (B, L_md, L_c + 1)
                probs = torch.softmax(logits, dim=-1)

                # Indices: [0, 1, 2, ...]
                # We use the dimension of logits to determine max possible rank
                max_rank = logits.size(-1)
                rank_indices = torch.arange(max_rank, device=self.device).float()

                # Expected Rank = sum(prob * index) -> (B, L_md)
                expected_ranks = (probs * rank_indices).sum(dim=-1)

                # 2. Reconstruct Order per Notebook
                batch_size = code_emb.size(0)

                for b in range(batch_size):
                    # Get notebook ID from dataset
                    nb_id = dataset.samples[current_idx + b]["id"]

                    if nb_id not in self.val_map:
                        continue

                    code_ids, md_ids = self.val_map[nb_id]

                    # Retrieve predicted scores for valid markdown cells
                    # md_mask[b] indicates valid cells.
                    # Note: md_ids length should match md_mask[b].sum()
                    num_md = len(md_ids)
                    scores = expected_ranks[b, :num_md].cpu().numpy()

                    # Build list of (cell_id, score)
                    cells_with_scores = []

                    # Code Cells: Fixed scores (0.5, 1.5, ...)
                    # If code sequence was truncated in dataset, we still have all code_ids from map.
                    # The model only sees up to MAX_ANCHOR_SEQ_LEN.
                    # We assign increasing scores to all code cells.
                    for i, cid in enumerate(code_ids):
                        cells_with_scores.append((cid, i + 0.5))

                    # Markdown Cells: Predicted scores
                    for i, cid in enumerate(md_ids):
                        cells_with_scores.append((cid, scores[i]))

                    # Sort by score to determine order
                    cells_with_scores.sort(key=lambda x: x[1])

                    # Extract ID sequence
                    pred_order = [x[0] for x in cells_with_scores]

                    predictions.append(pred_order)
                    ground_truths.append(self.val_gt[nb_id].split())

                current_idx += batch_size

        # Compute Metric
        kt_score = compute_kendall_tau(predictions, ground_truths)
        avg_loss = total_loss / len(dataloader)

        print(f"Validation Loss: {avg_loss:.6f}")
        print(f"Validation Kendall Tau: {kt_score:.6f}")

        return avg_loss, kt_score

    def generate_submission(self, dataloader, dataset):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        if self.test_map is None:
            print("Loading test metadata for reconstruction...")
            self.test_map = self._load_metadata_map("test")

        self.model.eval()
        submission_rows = []
        current_idx = 0

        print("Generating submission...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", mininterval=30):
                code_emb = batch["code_emb"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_emb = batch["md_emb"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)

                logits = self.model(code_emb, code_mask, md_emb, md_mask)
                probs = torch.softmax(logits, dim=-1)

                rank_indices = torch.arange(logits.size(-1), device=self.device).float()
                expected_ranks = (probs * rank_indices).sum(dim=-1)

                batch_size = code_emb.size(0)

                for b in range(batch_size):
                    nb_id = dataset.samples[current_idx + b]["id"]

                    if nb_id not in self.test_map:
                        # Should not happen
                        continue

                    code_ids, md_ids = self.test_map[nb_id]
                    num_md = len(md_ids)
                    scores = expected_ranks[b, :num_md].cpu().numpy()

                    cells_with_scores = []
                    for i, cid in enumerate(code_ids):
                        cells_with_scores.append((cid, i + 0.5))

                    for i, cid in enumerate(md_ids):
                        cells_with_scores.append((cid, scores[i]))

                    cells_with_scores.sort(key=lambda x: x[1])
                    pred_order = [x[0] for x in cells_with_scores]

                    # Format as space-delimited string
                    pred_string = " ".join(pred_order)
                    submission_rows.append({"id": nb_id, "cell_order": pred_string})

                current_idx += batch_size

        # Save to CSV
        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
