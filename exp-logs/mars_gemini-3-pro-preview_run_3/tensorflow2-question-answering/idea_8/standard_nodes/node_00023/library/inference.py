import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import load_glove_embeddings, tokenize_text
from library.data import get_vocabulary, get_test_candidates
from library.models import ANBoWRanker, ConvBiDAFReader


class TestDataset(Dataset):
    """
    Dataset for inference on test candidates.
    """

    def __init__(self, df, vocab):
        self.data = df
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q_tokens = tokenize_text(row["q_text"])
        c_tokens = tokenize_text(row["c_text"])

        q_ids = self.vocab.convert_tokens_to_ids(q_tokens, Config.MAX_Q_LEN)
        c_ids = self.vocab.convert_tokens_to_ids(c_tokens, Config.MAX_DOC_LEN)

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "candidate_idx": idx,  # To map back to dataframe
        }


class InferencePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"InferencePipeline initialized on device: {self.device}")

        # Load Resources
        self.vocab = get_vocabulary(load_cached_data=True)
        self.embedding_matrix = load_glove_embeddings(
            self.vocab.token_to_idx, Config.EMBEDDING_DIM, load_cached_data=True
        )

        # Initialize Models
        self.ranker = ANBoWRanker(embedding_matrix=self.embedding_matrix).to(
            self.device
        )
        self.reader = ConvBiDAFReader(embedding_matrix=self.embedding_matrix).to(
            self.device
        )

        # Load Weights
        self._load_checkpoint(self.ranker, Config.RANKER_MODEL_PATH)
        self._load_checkpoint(self.reader, Config.READER_MODEL_PATH)

        self.ranker.eval()
        self.reader.eval()

    def _load_checkpoint(self, model, path):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {path}")
        else:
            print(f"Warning: Checkpoint not found at {path}. Using random weights.")

    def _get_best_span(self, start_probs, end_probs, max_span_len=30):
        """
        Finds the best valid span (start <= end) and (end - start < max_span_len).
        Returns (start_idx, end_idx, score).
        Indices are relative to the candidate context.
        """
        best_score = -1.0
        best_start = 0
        best_end = 0

        # start_probs: (seq_len,)
        # end_probs: (seq_len,)
        seq_len = len(start_probs)

        # Naive O(N^2) search is fine for short sequences (MAX_DOC_LEN=512)
        # Optimization: only check valid windows
        for s in range(seq_len):
            # Pruning: if start prob is too low, skip
            if start_probs[s] < 0.01:
                continue

            for e in range(s, min(seq_len, s + max_span_len)):
                score = start_probs[s] * end_probs[e]
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end = e

        return best_start, best_end, best_score

    def run_inference(self, load_cached_data=True, batch_size=Config.BATCH_SIZE):
        print("Starting Inference Pipeline...")

        # 1. Load Candidates
        # Schema: example_id, q_text, c_text, c_idx, token_start, token_end
        candidates_df = get_test_candidates(load_cached_data=load_cached_data)

        if candidates_df.empty:
            print("No test candidates found. Generating empty submission.")
            self._generate_empty_submission()
            return

        # 2. Ranking Phase
        print("Ranking candidates...")
        dataset = TestDataset(candidates_df, self.vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        all_ranker_scores = []

        with torch.no_grad():
            for batch in dataloader:
                q_ids = batch["q_ids"].to(self.device)
                c_ids = batch["c_ids"].to(self.device)

                logits = self.ranker(q_ids, c_ids)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_ranker_scores.extend(probs)

        candidates_df["ranker_score"] = all_ranker_scores

        # 3. Selection Phase
        # For each example_id, keep only the candidate with the highest ranker score
        best_candidates_idx = candidates_df.groupby("example_id")[
            "ranker_score"
        ].idxmax()
        top_candidates_df = candidates_df.loc[best_candidates_idx].reset_index(
            drop=True
        )

        print(f"Selected top {len(top_candidates_df)} candidates for reading phase.")

        # 4. Reading Phase
        print("Extracting answers...")
        reader_dataset = TestDataset(top_candidates_df, self.vocab)
        reader_loader = DataLoader(
            reader_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        results = []

        with torch.no_grad():
            for i, batch in enumerate(reader_loader):
                q_ids = batch["q_ids"].to(self.device)
                c_ids = batch["c_ids"].to(self.device)

                # Get logits: (B, Seq_Len)
                start_logits, end_logits = self.reader(q_ids, c_ids)

                # Softmax
                start_probs = F.softmax(start_logits, dim=-1).cpu().numpy()
                end_probs = F.softmax(end_logits, dim=-1).cpu().numpy()

                # Process batch
                batch_indices = batch[
                    "candidate_idx"
                ].numpy()  # Indices into top_candidates_df (if we didn't shuffle)
                # Actually, TestDataset returns index relative to the dataframe passed to it.
                # Since shuffle=False, i * batch_size + j corresponds to row index in top_candidates_df

                current_batch_size = q_ids.size(0)

                for j in range(current_batch_size):
                    global_idx = i * batch_size + j
                    row = top_candidates_df.iloc[global_idx]

                    # Get best span
                    s_idx, e_idx, span_score = self._get_best_span(
                        start_probs[j], end_probs[j]
                    )

                    # Calculate final confidence
                    ranker_score = row["ranker_score"]
                    final_confidence = ranker_score * span_score

                    # Determine predictions based on threshold
                    long_pred = ""
                    short_pred = ""

                    if final_confidence >= Config.INFERENCE_THRESHOLD:
                        # Long Answer: The whole candidate paragraph
                        la_start = int(row["token_start"])
                        la_end = int(row["token_end"])
                        long_pred = f"{la_start}:{la_end}"

                        # Short Answer: Relative span -> Global indices
                        # Note: e_idx from model is inclusive index in candidate
                        # Submission requires exclusive end index?
                        # NQ format usually: start_token:end_token (exclusive)
                        # Model output e_idx is inclusive. So exclusive_end = e_idx + 1

                        sa_rel_start = s_idx
                        sa_rel_end_exclusive = e_idx + 1

                        sa_global_start = la_start + sa_rel_start
                        sa_global_end = la_start + sa_rel_end_exclusive

                        # Clip to candidate bounds just in case
                        sa_global_start = max(la_start, min(sa_global_start, la_end))
                        sa_global_end = max(sa_global_start, min(sa_global_end, la_end))

                        short_pred = f"{sa_global_start}:{sa_global_end}"

                    # Append results
                    # Format:
                    # - example_id_long, prediction
                    # - example_id_short, prediction

                    ex_id = str(row["example_id"])
                    results.append(
                        {"example_id": f"{ex_id}_long", "PredictionString": long_pred}
                    )
                    results.append(
                        {"example_id": f"{ex_id}_short", "PredictionString": short_pred}
                    )

        # 5. Save Submission
        self._save_submission(results)

    def _generate_empty_submission(self):
        """
        Generates a submission file with all empty predictions if no candidates are found.
        """
        metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        results = []
        for _, row in metadata.iterrows():
            ex_id = str(row["example_id"])
            results.append({"example_id": f"{ex_id}_long", "PredictionString": ""})
            results.append({"example_id": f"{ex_id}_short", "PredictionString": ""})
        self._save_submission(results)

    def _save_submission(self, results):
        """
        Saves the results list to CSV.
        """
        submission_df = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
