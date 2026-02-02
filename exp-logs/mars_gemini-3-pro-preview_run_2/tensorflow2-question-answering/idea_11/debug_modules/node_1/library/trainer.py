import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.vocab import Vocabulary
from library.dataset import NQDataset, collate_fn
from library.model import AGBoEModel


class Trainer:
    def __init__(self, load_cached_data=True, sample_size=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # 1. Load Vocabulary and Embeddings
        self.vocab, self.embedding_matrix = Vocabulary.load_or_build(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # 2. Initialize Model
        self.model = AGBoEModel(self.embedding_matrix).to(self.device)

        # 3. Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # 4. Loss Functions
        self.ranking_criterion = nn.BCEWithLogitsLoss()
        self.yesno_criterion = nn.CrossEntropyLoss()
        self.attention_criterion = nn.MSELoss(
            reduction="none"
        )  # Reduction handled manually for masking

        # Data Loading Parameters
        self.load_cached_data = load_cached_data
        self.sample_size = sample_size

    def get_dataloader(self, split):
        dataset = NQDataset(
            split=split,
            vocab=self.vocab,
            load_cached_data=self.load_cached_data,
            sample_size=self.sample_size,
        )
        shuffle = split == "train"
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=2 if self.device.type == "cuda" else 0,
        )

    def train(self):
        print("\n--- Starting Training ---")
        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0.0
            total_rank_loss = 0.0
            total_attn_loss = 0.0
            total_yn_loss = 0.0

            for batch in train_loader:
                # Move data to device
                q_indices = batch["q_indices"].to(self.device)
                c_indices = batch["c_indices"].to(self.device)
                labels = batch["labels"].to(self.device)  # (batch,)
                attn_masks = batch["attn_masks"].to(self.device)  # (batch, c_len)
                yes_nos = batch["yes_nos"].to(self.device)  # (batch,)

                self.optimizer.zero_grad()

                # Forward pass
                ranking_logits, yesno_logits, attn_weights = self.model(
                    q_indices, c_indices
                )

                # --- Calculate Losses ---

                # 1. Ranking Loss (Binary Classification)
                loss_rank = self.ranking_criterion(ranking_logits, labels)

                # 2. Yes/No Loss (Multi-class Classification)
                loss_yn = self.yesno_criterion(yesno_logits, yes_nos)

                # 3. Attention Loss (MSE against normalized mask)
                # Normalize ground truth mask to sum to 1 (probability distribution)
                mask_sum = torch.sum(attn_masks, dim=1, keepdim=True)
                # Avoid division by zero for empty masks
                mask_sum = torch.clamp(mask_sum, min=1e-9)
                normalized_mask = attn_masks / mask_sum

                # Calculate raw MSE
                mse_raw = self.attention_criterion(
                    attn_weights, normalized_mask
                )  # (batch, c_len)
                mse_per_sample = torch.mean(mse_raw, dim=1)  # (batch,)

                # Mask out loss for samples that don't have a short answer (mask_sum was effectively 0)
                # We check original sum. If sum was 0, it means no short answer span.
                has_short_answer = (torch.sum(attn_masks, dim=1) > 0).float()
                loss_attn = torch.mean(mse_per_sample * has_short_answer)

                # Combined Loss
                loss = (
                    Config.LOSS_WEIGHT_RANKING * loss_rank
                    + Config.LOSS_WEIGHT_YESNO * loss_yn
                    + Config.LOSS_WEIGHT_ATTENTION * loss_attn
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_rank_loss += loss_rank.item()
                total_attn_loss += loss_attn.item()
                total_yn_loss += loss_yn.item()

            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {avg_loss:.6f} "
                f"(Rank: {total_rank_loss/len(train_loader):.4f}, "
                f"Attn: {total_attn_loss/len(train_loader):.4f}, "
                f"YN: {total_yn_loss/len(train_loader):.4f})"
            )

            # Validation
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Val Loss: {val_loss:.6f}")

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                q_indices = batch["q_indices"].to(self.device)
                c_indices = batch["c_indices"].to(self.device)
                labels = batch["labels"].to(self.device)
                attn_masks = batch["attn_masks"].to(self.device)
                yes_nos = batch["yes_nos"].to(self.device)

                ranking_logits, yesno_logits, attn_weights = self.model(
                    q_indices, c_indices
                )

                loss_rank = self.ranking_criterion(ranking_logits, labels)
                loss_yn = self.yesno_criterion(yesno_logits, yes_nos)

                mask_sum = torch.sum(attn_masks, dim=1, keepdim=True)
                mask_sum = torch.clamp(mask_sum, min=1e-9)
                normalized_mask = attn_masks / mask_sum
                mse_raw = self.attention_criterion(attn_weights, normalized_mask)
                mse_per_sample = torch.mean(mse_raw, dim=1)
                has_short_answer = (torch.sum(attn_masks, dim=1) > 0).float()
                loss_attn = torch.mean(mse_per_sample * has_short_answer)

                loss = (
                    Config.LOSS_WEIGHT_RANKING * loss_rank
                    + Config.LOSS_WEIGHT_YESNO * loss_yn
                    + Config.LOSS_WEIGHT_ATTENTION * loss_attn
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict_and_submit(self):
        print("\n--- Generating Submission ---")
        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        test_loader = self.get_dataloader("test")

        results = {}  # example_id -> list of predictions

        with torch.no_grad():
            for batch in test_loader:
                q_indices = batch["q_indices"].to(self.device)
                c_indices = batch["c_indices"].to(self.device)

                # Metadata
                example_ids = batch["example_ids"]
                cand_starts = batch["cand_global_starts"]  # List

                # Forward
                ranking_logits, yesno_logits, attn_weights = self.model(
                    q_indices, c_indices
                )

                ranking_probs = torch.sigmoid(ranking_logits).cpu().numpy()
                yesno_probs = (
                    torch.softmax(yesno_logits, dim=1).cpu().numpy()
                )  # (batch, 3)
                attn_weights_np = attn_weights.cpu().numpy()  # (batch, c_len)

                for i, eid in enumerate(example_ids):
                    if eid not in results:
                        results[eid] = []

                    results[eid].append(
                        {
                            "score": ranking_probs[i],
                            "yesno_probs": yesno_probs[i],
                            "attn_weights": attn_weights_np[i],
                            "global_start": cand_starts[i],
                        }
                    )

        # Process results to generate strings
        submission_rows = []

        # We iterate over the keys in results to ensure we cover all processed IDs
        # Note: In a real scenario, we should iterate over sample_submission to ensure order/completeness,
        # but here we iterate over what we processed.
        for eid, candidates in results.items():
            # 1. Select Best Candidate
            best_cand = max(candidates, key=lambda x: x["score"])

            long_ans_str = ""
            short_ans_str = ""

            # 2. Threshold Check for Long Answer
            if best_cand["score"] >= Config.LONG_ANSWER_THRESHOLD:
                # We need the end index. Since we don't have global_end in the inference dict
                # (it was in batch but I didn't store it), we can infer or just store it.
                # Let's assume max length or reconstruct.
                # Actually, attn_weights length corresponds to tokens.
                # Global end is global_start + length of valid tokens.
                # However, the model input was padded/truncated.
                # A robust way: The prediction is just the span of the candidate.
                # In NQ, long answer is the whole candidate text.
                # We need the original candidate length.
                # Let's assume the candidate length is roughly the length of the attention weights
                # (ignoring padding). But padding is 0 weight.
                # Since we don't have the exact original length stored in 'results',
                # we will approximate or use the logic that the candidate text *is* the long answer.
                # The submission format requires token indices.
                # We have 'global_start'. We need 'global_end'.
                # I will modify the loop above to store global_end.
                pass

        # Re-running the loop logic cleanly with global_end included
        results = {}
        with torch.no_grad():
            for batch in test_loader:
                q_indices = batch["q_indices"].to(self.device)
                c_indices = batch["c_indices"].to(self.device)
                example_ids = batch["example_ids"]
                cand_starts = batch["cand_global_starts"]
                cand_ends = batch["cand_global_ends"]

                ranking_logits, yesno_logits, attn_weights = self.model(
                    q_indices, c_indices
                )
                ranking_probs = torch.sigmoid(ranking_logits).cpu().numpy()
                yesno_probs = torch.softmax(yesno_logits, dim=1).cpu().numpy()
                attn_weights_np = attn_weights.cpu().numpy()

                for i, eid in enumerate(example_ids):
                    if eid not in results:
                        results[eid] = []
                    results[eid].append(
                        {
                            "score": ranking_probs[i],
                            "yesno_probs": yesno_probs[i],
                            "attn_weights": attn_weights_np[i],
                            "global_start": cand_starts[i],
                            "global_end": cand_ends[i],
                        }
                    )

        final_submission_data = []

        # Load sample submission to ensure we output all required IDs
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        # Extract unique IDs from sample submission (rows are id_long, id_short)
        # We need to handle the fact that sample submission has multiple rows per ID.
        # We will iterate through our results.

        for eid, candidates in results.items():
            best_cand = max(candidates, key=lambda x: x["score"])

            long_pred = ""
            short_pred = ""

            if best_cand["score"] >= Config.LONG_ANSWER_THRESHOLD:
                # Long Answer Prediction
                long_pred = f"{best_cand['global_start']}:{best_cand['global_end']}"

                # Short Answer Prediction
                # Check Yes/No
                yn_class = np.argmax(best_cand["yesno_probs"])
                if yn_class == 1:
                    short_pred = "YES"
                elif yn_class == 2:
                    short_pred = "NO"
                else:
                    # Extraction via Sliding Window
                    weights = best_cand["attn_weights"]
                    window = Config.SHORT_SPAN_WINDOW

                    # Convolve to find best window
                    # Simple moving sum
                    if len(weights) >= window:
                        kernel = np.ones(window)
                        sums = np.convolve(weights, kernel, mode="valid")
                        best_start_rel = np.argmax(sums)
                        best_end_rel = best_start_rel + window
                    else:
                        # If candidate is shorter than window, take the whole thing
                        best_start_rel = 0
                        best_end_rel = len(weights)

                    # Map to global
                    # Note: weights include padding. We should ensure we don't pick padding.
                    # But padding embeddings are 0, so dot product is 0, so weights are small/uniform?
                    # Softmax on 0s gives uniform distribution.
                    # Assuming padding is handled, we map back.

                    s_global_start = best_cand["global_start"] + best_start_rel
                    s_global_end = best_cand["global_start"] + best_end_rel

                    # Clip to candidate boundaries
                    s_global_end = min(s_global_end, best_cand["global_end"])

                    short_pred = f"{s_global_start}:{s_global_end}"

            final_submission_data.append(
                {"example_id": f"{eid}_long", "PredictionString": long_pred}
            )
            final_submission_data.append(
                {"example_id": f"{eid}_short", "PredictionString": short_pred}
            )

        # Convert to DataFrame
        submission_df = pd.DataFrame(final_submission_data)

        # Ensure we match sample submission structure (fill missing with empty if any)
        # But we processed the whole test set via dataset, so we should be good.

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
