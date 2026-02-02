import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import compute_kendall_tau


class Engine:
    def __init__(self, model, device, optimizer=None):
        """
        Initializes the training engine.

        Args:
            model: The PyTorch model (DCAN).
            device: The device to run on (cuda/cpu).
            optimizer: The optimizer (optional, for training).
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.config = Config()
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, data_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            # Move inputs to device
            code_emb = batch["code_embeddings"].to(self.device)
            md_emb = batch["markdown_embeddings"].to(self.device)
            labels = batch["labels"].to(self.device)
            code_lens = batch["code_lens"].to(self.device)
            md_lens = batch["md_lens"].to(self.device)

            if self.optimizer:
                self.optimizer.zero_grad()

            # Forward pass
            # logits shape: (Batch, Num_MD, Num_Code + 1)
            logits = self.model(code_emb, md_emb, code_lens, md_lens)

            # Flatten inputs for CrossEntropyLoss
            # Logits: (Batch * Num_MD, Num_Code + 1)
            # Labels: (Batch * Num_MD)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            if self.optimizer:
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f}")
        return avg_loss

    def evaluate(self, data_loader, raw_df):
        """
        Evaluates the model on the validation set using Kendall Tau.

        Args:
            data_loader: Validation DataLoader.
            raw_df: DataFrame containing metadata (code_ids, markdown_ids, cell_order).
        """
        self.model.eval()

        # Create lookup for notebook structure
        nb_meta = raw_df.set_index("id")[["code_ids", "markdown_ids"]].to_dict("index")

        preds_list = []

        with torch.no_grad():
            for batch in data_loader:
                code_emb = batch["code_embeddings"].to(self.device)
                md_emb = batch["markdown_embeddings"].to(self.device)
                code_lens = batch["code_lens"].to(self.device)
                md_lens = batch["md_lens"].to(self.device)
                ids = batch["ids"]

                logits = self.model(code_emb, md_emb, code_lens, md_lens)

                # Calculate Soft Ranks (Expected Index)
                probs = torch.softmax(logits, dim=2)  # (B, M, C+1)
                max_cls = probs.size(2)
                indices = (
                    torch.arange(max_cls, device=self.device).float().view(1, 1, -1)
                )
                expected_ranks = (
                    torch.sum(probs * indices, dim=2).cpu().numpy()
                )  # (B, M)

                # Reconstruct Order
                for i, nb_id in enumerate(ids):
                    if nb_id not in nb_meta:
                        continue

                    curr_code_ids = nb_meta[nb_id]["code_ids"]
                    curr_md_ids = nb_meta[nb_id]["markdown_ids"]

                    # Get valid number of markdown cells processed
                    curr_md_len = md_lens[i].item()
                    num_md = min(len(curr_md_ids), curr_md_len)

                    ranks = expected_ranks[i, :num_md]

                    # Assign ranks: Code cells = index + 0.5
                    cell_rank_pairs = []
                    for c_idx, c_id in enumerate(curr_code_ids):
                        cell_rank_pairs.append((c_id, c_idx + 0.5))

                    # Markdown cells = predicted expected rank
                    for m_idx in range(num_md):
                        cell_rank_pairs.append((curr_md_ids[m_idx], ranks[m_idx]))

                    # Handle truncated markdown cells (append to end)
                    if len(curr_md_ids) > num_md:
                        for m_idx in range(num_md, len(curr_md_ids)):
                            cell_rank_pairs.append(
                                (curr_md_ids[m_idx], len(curr_code_ids) + 100.0 + m_idx)
                            )

                    # Sort by rank
                    cell_rank_pairs.sort(key=lambda x: x[1])

                    # Construct order string
                    pred_order = " ".join([cid for cid, r in cell_rank_pairs])
                    preds_list.append({"id": nb_id, "cell_order": pred_order})

        df_pred = pd.DataFrame(preds_list)
        df_gt = raw_df[["id", "cell_order"]].copy()

        score = compute_kendall_tau(df_pred, df_gt)
        print(f"Validation Kendall Tau: {score:.6f}")
        return score

    def fit(self, train_loader, val_loader, train_df, val_df, epochs, patience):
        """
        Runs the full training loop with Early Stopping.
        """
        best_score = -1.0
        patience_counter = 0

        for epoch in range(epochs):
            self.train_one_epoch(train_loader, epoch)
            score = self.evaluate(val_loader, val_df)

            if score > best_score:
                best_score = score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"New best model saved with score: {best_score:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self, data_loader, raw_df):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        self.model.eval()
        nb_meta = raw_df.set_index("id")[["code_ids", "markdown_ids"]].to_dict("index")
        preds_list = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in data_loader:
                code_emb = batch["code_embeddings"].to(self.device)
                md_emb = batch["markdown_embeddings"].to(self.device)
                code_lens = batch["code_lens"].to(self.device)
                md_lens = batch["md_lens"].to(self.device)
                ids = batch["ids"]

                logits = self.model(code_emb, md_emb, code_lens, md_lens)

                probs = torch.softmax(logits, dim=2)
                max_cls = probs.size(2)
                indices = (
                    torch.arange(max_cls, device=self.device).float().view(1, 1, -1)
                )
                expected_ranks = torch.sum(probs * indices, dim=2).cpu().numpy()

                for i, nb_id in enumerate(ids):
                    if nb_id not in nb_meta:
                        continue
                    curr_code_ids = nb_meta[nb_id]["code_ids"]
                    curr_md_ids = nb_meta[nb_id]["markdown_ids"]
                    curr_md_len = md_lens[i].item()
                    num_md = min(len(curr_md_ids), curr_md_len)

                    ranks = expected_ranks[i, :num_md]

                    cell_rank_pairs = []
                    for c_idx, c_id in enumerate(curr_code_ids):
                        cell_rank_pairs.append((c_id, c_idx + 0.5))
                    for m_idx in range(num_md):
                        cell_rank_pairs.append((curr_md_ids[m_idx], ranks[m_idx]))
                    if len(curr_md_ids) > num_md:
                        for m_idx in range(num_md, len(curr_md_ids)):
                            cell_rank_pairs.append(
                                (curr_md_ids[m_idx], len(curr_code_ids) + 100.0 + m_idx)
                            )

                    cell_rank_pairs.sort(key=lambda x: x[1])
                    pred_order = " ".join([cid for cid, r in cell_rank_pairs])
                    preds_list.append({"id": nb_id, "cell_order": pred_order})

        df_sub = pd.DataFrame(preds_list)
        df_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
        return df_sub
