import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from library import config
from library import text_utils
from library import data_factory
from library.ranker_net import KMaxInteractionRanker
from library.reader_net import HighwayCoAttentionReader

# -----------------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------------


class InferenceDataset(Dataset):
    """
    Simple Dataset wrapper for inference tensors.
    """

    def __init__(self, q_indices, ctx_indices):
        self.q_indices = q_indices
        self.ctx_indices = ctx_indices

    def __len__(self):
        return len(self.q_indices)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.q_indices[idx], dtype=torch.long),
            torch.tensor(self.ctx_indices[idx], dtype=torch.long),
        )


# -----------------------------------------------------------------------------
# Inference Pipeline
# -----------------------------------------------------------------------------


class InferencePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = None
        self.embeddings = None
        self.ranker = None
        self.reader = None

    def load_resources(self):
        """
        Loads vocabulary and embeddings.
        """
        print("Loading vocabulary and embeddings...")
        # Load vocab (force build/load via text_utils logic)
        self.vocab = text_utils.build_vocab(load_cached_data=True)

        # Load embeddings
        self.embeddings = text_utils.load_embeddings(self.vocab, load_cached_data=True)

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Embedding shape: {self.embeddings.shape}")

    def load_models(self):
        """
        Initializes and loads trained models.
        """
        if self.vocab is None:
            self.load_resources()

        vocab_size = len(self.vocab)

        print("Loading Ranker model...")
        self.ranker = KMaxInteractionRanker(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            pretrained_embeddings=self.embeddings,
            k=config.K_MAX,
            hidden_dim=config.HIDDEN_DIM,
            dropout_rate=config.DROPOUT_RATE,
        )

        if os.path.exists(config.RANKER_MODEL_PATH):
            state_dict = torch.load(config.RANKER_MODEL_PATH, map_location=self.device)
            self.ranker.load_state_dict(state_dict)
            print(f"Ranker weights loaded from {config.RANKER_MODEL_PATH}")
        else:
            print(
                f"Warning: Ranker checkpoint not found at {config.RANKER_MODEL_PATH}. Using random weights."
            )

        self.ranker.to(self.device)
        self.ranker.eval()

        print("Loading Reader model...")
        self.reader = HighwayCoAttentionReader(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            pretrained_embeddings=self.embeddings,
            hidden_dim=config.HIDDEN_DIM,
            num_highway_layers=config.HIGHWAY_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
        )

        if os.path.exists(config.READER_MODEL_PATH):
            state_dict = torch.load(config.READER_MODEL_PATH, map_location=self.device)
            self.reader.load_state_dict(state_dict)
            print(f"Reader weights loaded from {config.READER_MODEL_PATH}")
        else:
            print(
                f"Warning: Reader checkpoint not found at {config.READER_MODEL_PATH}. Using random weights."
            )

        self.reader.to(self.device)
        self.reader.eval()

    def _get_best_span(self, start_logits, end_logits, max_span_len=30):
        """
        Finds the optimal span (start, end) that maximizes the joint probability.
        """
        # Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=-1)  # (Seq_Len,)
        end_probs = F.softmax(end_logits, dim=-1)  # (Seq_Len,)

        # Create score matrix: score[i, j] = P_start[i] * P_end[j]
        # Shape: (Seq_Len, Seq_Len)
        score_mat = torch.ger(start_probs, end_probs)

        # Create mask for valid spans
        # 1. start <= end (upper triangular)
        # 2. end - start < max_span_len
        seq_len = start_logits.size(0)

        # Upper triangular mask (start <= end)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device))

        # Length constraint mask
        # We can implement this by zeroing out elements too far from diagonal
        # Or simpler iteration since seq_len is small (MAX_DOC_LEN=256)

        # Apply basic validity mask
        score_mat = score_mat * mask

        # Apply length constraint manually or via band mask
        # For simplicity and speed on small matrices, we can iterate or use band part
        # Here we just zero out invalid length spans
        for i in range(seq_len):
            # Invalid if end index > i + max_span_len
            if i + max_span_len < seq_len:
                score_mat[i, i + max_span_len :] = 0

        # Find max
        max_score = score_mat.max()
        flat_idx = score_mat.argmax()

        best_start = (flat_idx // seq_len).item()
        best_end = (flat_idx % seq_len).item()

        return best_start, best_end, max_score.item()

    def run_inference(self):
        """
        Main execution method for generating predictions on the test set.
        """
        # 1. Setup
        if self.ranker is None:
            self.load_models()

        # 2. Prepare Data (Candidates for Ranking)
        # This generates a DF with all candidates for all test questions
        print("Processing test inputs...")
        ranker_inputs_df = data_factory.process_test_ranker_inputs(
            config.TEST_METADATA_PATH, self.vocab, load_cached_data=True
        )

        if ranker_inputs_df.empty:
            print("No test data found or processed.")
            self._generate_empty_submission()
            return

        # 3. Rank Candidates
        print("Ranking candidates...")
        # Create dataset and loader
        # We need q_indices and cand_indices
        q_seqs = ranker_inputs_df["q_indices"].tolist()
        c_seqs = ranker_inputs_df["cand_indices"].tolist()

        ranker_dataset = InferenceDataset(q_seqs, c_seqs)
        ranker_loader = DataLoader(
            ranker_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        all_scores = []
        with torch.no_grad():
            for q_batch, c_batch in ranker_loader:
                q_batch = q_batch.to(self.device)
                c_batch = c_batch.to(self.device)

                scores = self.ranker(q_batch, c_batch)
                all_scores.extend(scores.cpu().numpy().tolist())

        ranker_inputs_df["rank_score"] = all_scores

        # 4. Select Best Candidate per Question
        # Group by example_id and find the row with max rank_score
        print("Selecting best candidates...")
        # idxmax returns the index of the max value
        best_candidates_idx = ranker_inputs_df.groupby("example_id")[
            "rank_score"
        ].idxmax()
        best_candidates_df = ranker_inputs_df.loc[best_candidates_idx].reset_index(
            drop=True
        )

        # 5. Extract Short Answers using Reader
        print("Extracting short answers...")
        # Prepare reader inputs from the best candidates
        # Note: We reuse the indices. Ranker and Reader use same max lengths in this setup.
        q_seqs_best = best_candidates_df["q_indices"].tolist()
        c_seqs_best = best_candidates_df["cand_indices"].tolist()

        reader_dataset = InferenceDataset(q_seqs_best, c_seqs_best)
        reader_loader = DataLoader(
            reader_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        predictions = {}  # Map example_id -> {long_pred, short_pred}

        cursor = 0
        with torch.no_grad():
            for q_batch, c_batch in reader_loader:
                q_batch = q_batch.to(self.device)
                c_batch = c_batch.to(self.device)

                start_logits_batch, end_logits_batch = self.reader(q_batch, c_batch)

                batch_size = q_batch.size(0)

                for i in range(batch_size):
                    # Get results for this sample
                    start_logits = start_logits_batch[i]
                    end_logits = end_logits_batch[i]

                    # Find best span
                    best_s, best_e, confidence = self._get_best_span(
                        start_logits, end_logits
                    )

                    # Get metadata from dataframe
                    row = best_candidates_df.iloc[cursor + i]
                    ex_id = row["example_id"]
                    cand_start_token = row["start_token"]

                    # Determine predictions based on threshold
                    long_ans_str = ""
                    short_ans_str = ""

                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        # Long Answer: The whole candidate paragraph
                        # Format: start:end (global indices)
                        la_start = cand_start_token
                        la_end = row["end_token"]
                        long_ans_str = f"{la_start}:{la_end}"

                        # Short Answer: Span within candidate
                        # Map relative index to global index
                        # Note: best_s is relative to the start of the candidate
                        sa_start_global = cand_start_token + best_s
                        sa_end_global = (
                            cand_start_token + best_e + 1
                        )  # +1 because submission format usually expects exclusive end or inclusive?
                        # NQ evaluation usually expects token spans.
                        # The sample submission format is "start:end".
                        # In NQ, annotations are start (inclusive) and end (exclusive).
                        # Our text_utils segment_document uses exclusive end.
                        # Our reader predicts inclusive end index relative to paragraph.
                        # So global exclusive end = cand_start + best_e + 1.

                        short_ans_str = f"{sa_start_global}:{sa_end_global}"

                    predictions[ex_id] = {"long": long_ans_str, "short": short_ans_str}

                cursor += batch_size

        # 6. Generate Submission File
        self._format_submission(predictions)

    def _generate_empty_submission(self):
        """
        Generates a submission file with all nulls if processing fails.
        """
        print("Generating empty submission...")
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
        sample_sub["PredictionString"] = config.NULL_PREDICTION_STRING
        sample_sub.to_csv(config.SUBMISSION_PATH, index=False)

    def _format_submission(self, predictions):
        """
        Formats the predictions dictionary into the final CSV.
        """
        print("Formatting submission...")
        # Load sample submission to get all required IDs
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

        final_preds = []

        for _, row in sample_sub.iterrows():
            # ID format: {example_id}_{type} (e.g., -12345_long)
            full_id = row["example_id"]
            example_id = full_id.rsplit("_", 1)[0]
            pred_type = full_id.rsplit("_", 1)[1]  # long or short

            pred_str = config.NULL_PREDICTION_STRING

            if example_id in predictions:
                if pred_type == "long":
                    pred_str = predictions[example_id]["long"]
                elif pred_type == "short":
                    pred_str = predictions[example_id]["short"]

            final_preds.append(pred_str)

        sample_sub["PredictionString"] = final_preds
        sample_sub.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
