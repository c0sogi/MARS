import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.data_utils import HTMLSegmenter, Vocabulary, load_embeddings
from library.models import EarlyFusionRanker, DynamicKernelReader


class InferencePipeline:
    """
    Orchestrates the evaluation pipeline using trained Ranker and Reader models.
    """

    def __init__(self, load_cached_data=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmenter = HTMLSegmenter()
        self.vocab = Vocabulary()

        # Load Vocabulary
        if os.path.exists(Config.VOCAB_PATH):
            self.vocab.load(Config.VOCAB_PATH)
        else:
            print(
                f"Warning: Vocabulary not found at {Config.VOCAB_PATH}. Inference may fail."
            )

        # Load Embeddings
        self.embeddings = load_embeddings(
            self.vocab, Config.EMBEDDING_DIM, load_cached_data=load_cached_data
        )

        # Initialize and Load Models
        self.ranker = EarlyFusionRanker(embedding_matrix=self.embeddings).to(
            self.device
        )
        self.reader = DynamicKernelReader(embedding_matrix=self.embeddings).to(
            self.device
        )

        self._load_model_weights(self.ranker, Config.RANKER_MODEL_PATH)
        self._load_model_weights(self.reader, Config.READER_MODEL_PATH)

        self.ranker.eval()
        self.reader.eval()

    def _load_model_weights(self, model, path):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {path}. Using random weights."
            )

    def segment_document(self, document_text):
        """Wraps the HTMLSegmenter to get candidate paragraphs."""
        return self.segmenter.segment(document_text)

    def rank_candidates(self, question_tokens, candidates):
        """
        Scores list of candidates using EarlyFusionRanker.
        Returns: list of floats (scores)
        """
        if not candidates:
            return []

        # Prepare batch
        input_ids_list = []
        q_ids = self.vocab.encode(question_tokens, Config.MAX_QUESTION_LEN)
        sep_id = self.vocab.token2idx[Config.SEP_TOKEN]
        pad_id = self.vocab.token2idx[Config.PAD_TOKEN]

        for cand in candidates:
            p_ids = self.vocab.encode(cand["tokens"], Config.MAX_PARAGRAPH_LEN)
            combined = q_ids + [sep_id] + p_ids

            # Pad/Truncate
            if len(combined) < Config.MAX_RANKER_SEQ_LEN:
                combined += [pad_id] * (Config.MAX_RANKER_SEQ_LEN - len(combined))
            else:
                combined = combined[: Config.MAX_RANKER_SEQ_LEN]

            input_ids_list.append(combined)

        # Convert to tensor
        input_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.ranker(input_tensor)
            scores = torch.sigmoid(logits).cpu().numpy().tolist()

        return scores

    def extract_answer(self, question_tokens, candidate):
        """
        Extracts short answer span from the candidate paragraph.
        Returns: (rel_start, rel_end, score)
        """
        q_ids = self.vocab.encode(question_tokens, Config.MAX_QUESTION_LEN)
        ctx_ids = self.vocab.encode(candidate["tokens"], Config.MAX_PARAGRAPH_LEN)

        q_tensor = torch.tensor([q_ids], dtype=torch.long).to(self.device)
        ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            start_logits, end_logits = self.reader(q_tensor, ctx_tensor)

            # Get probabilities
            start_probs = torch.softmax(start_logits, dim=1)[0]
            end_probs = torch.softmax(end_logits, dim=1)[0]

        # Find best valid span
        best_score = -1.0
        best_span = (0, 0)

        # Simple heuristic search for best span
        # Limit max span length to reasonable size (e.g., 30 tokens) to speed up
        max_span_len = 30
        seq_len = len(ctx_ids)

        # Convert to numpy for loop
        start_probs = start_probs.cpu().numpy()
        end_probs = end_probs.cpu().numpy()

        # Get top K start and end indices to reduce complexity
        top_k = 10
        start_indices = np.argsort(start_probs)[-top_k:]
        end_indices = np.argsort(end_probs)[-top_k:]

        for s_idx in start_indices:
            for e_idx in end_indices:
                if s_idx <= e_idx and (e_idx - s_idx) < max_span_len:
                    score = start_probs[s_idx] * end_probs[e_idx]
                    if score > best_score:
                        best_score = score
                        best_span = (s_idx, e_idx)

        return best_span[0], best_span[1], best_score

    def predict_single(self, example_data):
        """
        Runs pipeline for a single example.
        Returns: (long_ans_str, short_ans_str)
        """
        doc_text = example_data.get("document_text", "")
        q_text = example_data.get("question_text", "")
        q_tokens = q_text.split()

        # 1. Segment
        candidates = self.segment_document(doc_text)
        if not candidates:
            return "", ""

        # 2. Rank
        scores = self.rank_candidates(q_tokens, candidates)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_cand = candidates[best_idx]

        # Threshold Long Answer
        if best_score < Config.CONFIDENCE_THRESHOLD:
            return "", ""

        # Construct Long Answer String (token indices)
        long_ans_str = f"{best_cand['start_token']}:{best_cand['end_token']}"

        # 3. Extract Short Answer
        rel_start, rel_end, short_score = self.extract_answer(q_tokens, best_cand)

        # Threshold Short Answer
        # Note: Short answer score is conditional on the long answer being correct.
        # We can combine scores or just threshold the reader score.
        if short_score < Config.CONFIDENCE_THRESHOLD:
            short_ans_str = ""
        else:
            # Convert relative to absolute
            abs_start = best_cand["start_token"] + rel_start
            # End index in submission is usually exclusive (start:end)
            # The reader predicts inclusive end index relative to paragraph.
            # So absolute exclusive end = start + rel_end + 1
            abs_end = best_cand["start_token"] + rel_end + 1
            short_ans_str = f"{abs_start}:{abs_end}"

        return long_ans_str, short_ans_str

    def run_inference(self, load_cached_data=True):
        """
        Main execution method. Generates submission.csv.
        """
        Config.ensure_directories()

        # Determine test file path
        test_metadata_path = Config.TEST_METADATA_PATH
        if not os.path.exists(test_metadata_path):
            print(
                f"Test metadata not found at {test_metadata_path}. Cannot run inference."
            )
            return

        metadata_df = pd.read_csv(test_metadata_path)

        # We need to read the actual JSONL file.
        # Assuming all test examples come from the same file listed in metadata.
        if metadata_df.empty:
            print("Test metadata is empty.")
            return

        source_file = metadata_df.iloc[0]["file_path"]
        full_source_path = os.path.join(Config.INPUT_DIR, source_file)

        results = []

        print(f"Running inference on {len(metadata_df)} examples from {source_file}...")

        with open(full_source_path, "rb") as f:
            for _, row in metadata_df.iterrows():
                ex_id = row["example_id"]
                offset = row["byte_offset"]

                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))

                    # Retrieve authoritative ID from raw data (Cite debug_lesson_1)
                    raw_id = str(data.get("example_id", ex_id))

                    # Sanitize ID to prevent CSV injection (Cite debug_lesson_4)
                    safe_id = (
                        raw_id.strip()
                        .replace(",", "")
                        .replace("\n", "")
                        .replace("\r", "")
                    )

                    long_pred, short_pred = self.predict_single(data)

                    # Append rows for submission
                    results.append(
                        {"example_id": f"{safe_id}_long", "PredictionString": long_pred}
                    )
                    results.append(
                        {
                            "example_id": f"{safe_id}_short",
                            "PredictionString": short_pred,
                        }
                    )

                except json.JSONDecodeError:
                    # Fallback for corrupt lines
                    # Sanitize metadata ID as fallback
                    safe_id = (
                        str(ex_id)
                        .strip()
                        .replace(",", "")
                        .replace("\n", "")
                        .replace("\r", "")
                    )
                    results.append(
                        {"example_id": f"{safe_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{safe_id}_short", "PredictionString": ""}
                    )

        # Save Submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
