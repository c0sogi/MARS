import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.models import SiameseRanker, ConditionalReader
from library.data_processing import get_test_data_processor


class InferencePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference initialized on device: {self.device}")

        # Load Processor and Vocab
        self.processor = get_test_data_processor()
        self.vocab = self.processor.vocab
        self.text_processor = self.processor.text_processor

        # Load Models
        self.ranker = self._load_ranker()
        self.reader = self._load_reader()

    def _load_ranker(self):
        model = SiameseRanker(vocab_size=len(self.vocab)).to(self.device)
        if os.path.exists(Config.RANKER_MODEL_PATH):
            state_dict = torch.load(Config.RANKER_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded Ranker from {Config.RANKER_MODEL_PATH}")
        else:
            print("Warning: Ranker checkpoint not found. Using initialized weights.")
        model.eval()
        return model

    def _load_reader(self):
        model = ConditionalReader(vocab_size=len(self.vocab)).to(self.device)
        if os.path.exists(Config.READER_MODEL_PATH):
            state_dict = torch.load(Config.READER_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded Reader from {Config.READER_MODEL_PATH}")
        else:
            print("Warning: Reader checkpoint not found. Using initialized weights.")
        model.eval()
        return model

    def _prepare_tensor(self, tokens, max_len):
        """Encodes, truncates, and pads a token list into a tensor."""
        indices = self.vocab.encode(tokens)
        indices = indices[:max_len]
        pad_len = max_len - len(indices)
        indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * pad_len
        return torch.tensor([indices], dtype=torch.long).to(self.device)

    def _get_best_span(self, start_probs, end_probs, max_span_len=30):
        """Finds the optimal start and end indices maximizing joint probability."""
        best_score = -1.0
        best_start = 0
        best_end = 0

        # Convert to list for simple iteration
        s_probs = start_probs.tolist()
        e_probs = end_probs.tolist()
        seq_len = len(s_probs)

        # Greedy search with length constraint
        # Note: end index in loop is inclusive
        for i in range(seq_len):
            for j in range(i, min(seq_len, i + max_span_len)):
                score = s_probs[i] * e_probs[j]
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j

        return best_start, best_end, best_score

    def generate_predictions(self, sample_size=None):
        """
        Generates predictions for the test set and saves to submission.csv.
        Args:
            sample_size (int, optional): Number of samples to process for debugging.
        """
        print("Starting prediction generation...")

        # Load test metadata
        if not os.path.exists(Config.TEST_METADATA):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA}"
            )

        test_meta = pd.read_csv(Config.TEST_METADATA)
        if sample_size:
            test_meta = test_meta.head(sample_size)

        results = []

        # Open file once
        with open(Config.TEST_FILE, "rb") as f:
            for _, row in tqdm(
                test_meta.iterrows(), total=len(test_meta), disable=True
            ):
                example_id = str(row["example_id"])
                offset = row["byte_offset"]

                # Read sample
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                # 1. Preprocess Question
                q_text = data["question_text"]
                q_tokens = self.text_processor.tokenize(q_text)
                q_tensor = self._prepare_tensor(q_tokens, Config.MAX_Q_LEN)

                # 2. Preprocess Candidates
                doc_text = data["document_text"]
                doc_tokens = doc_text.split()
                candidates = data["long_answer_candidates"]

                if not candidates:
                    # No candidates, return empty
                    results.append(
                        {"example_id": f"{example_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{example_id}_short", "PredictionString": ""}
                    )
                    continue

                # Prepare batch for Ranker
                # Note: For very large docs, we might need to batch this loop.
                # NQ candidates usually < 100, fitting in memory.
                cand_tensors = []
                cand_clean_tokens_list = []
                cand_maps = []
                valid_cand_indices = []

                for idx, cand in enumerate(candidates):
                    raw_span = doc_tokens[cand["start_token"] : cand["end_token"]]
                    clean_tokens, idx_map = self.text_processor.clean_and_map_indices(
                        raw_span
                    )

                    if not clean_tokens:
                        continue

                    cand_clean_tokens_list.append(clean_tokens)
                    cand_maps.append(idx_map)
                    valid_cand_indices.append(idx)

                    # Create tensor (no batch dim yet, will stack)
                    indices = self.vocab.encode(clean_tokens)
                    indices = indices[: Config.MAX_CTX_LEN]
                    pad_len = Config.MAX_CTX_LEN - len(indices)
                    indices += [self.vocab.token_to_idx[Config.PAD_TOKEN]] * pad_len
                    cand_tensors.append(torch.tensor(indices, dtype=torch.long))

                if not cand_tensors:
                    results.append(
                        {"example_id": f"{example_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{example_id}_short", "PredictionString": ""}
                    )
                    continue

                # Stack for batch inference
                ctx_batch = torch.stack(cand_tensors).to(self.device)
                q_batch = q_tensor.repeat(len(ctx_batch), 1)

                # 3. Rank Candidates
                with torch.no_grad():
                    scores = self.ranker(q_batch, ctx_batch)

                best_score, best_idx_in_batch = torch.max(scores, dim=0)
                best_score = best_score.item()

                # Threshold Check
                if best_score < Config.RANKER_THRESHOLD:
                    results.append(
                        {"example_id": f"{example_id}_long", "PredictionString": ""}
                    )
                    results.append(
                        {"example_id": f"{example_id}_short", "PredictionString": ""}
                    )
                    continue

                # Retrieve best candidate info
                best_cand_original_idx = valid_cand_indices[best_idx_in_batch]
                best_cand_struct = candidates[best_cand_original_idx]
                best_clean_tokens = cand_clean_tokens_list[best_idx_in_batch]
                best_map = cand_maps[best_idx_in_batch]

                # Format Long Answer
                long_ans_str = (
                    f"{best_cand_struct['start_token']}:{best_cand_struct['end_token']}"
                )

                # 4. Extract Short Answer
                # Prepare input for Reader (single sample)
                q_reader_in = q_tensor  # [1, Q_LEN]
                ctx_reader_in = ctx_batch[best_idx_in_batch].unsqueeze(
                    0
                )  # [1, CTX_LEN]

                with torch.no_grad():
                    start_logits, end_logits = self.reader(q_reader_in, ctx_reader_in)

                # Mask padding in logits to avoid selecting pads
                # Actual length of clean tokens (before padding)
                actual_len = min(len(best_clean_tokens), Config.MAX_CTX_LEN)

                # Softmax
                start_probs = F.softmax(start_logits[0, :actual_len], dim=0)
                end_probs = F.softmax(end_logits[0, :actual_len], dim=0)

                pred_start_clean, pred_end_clean, span_score = self._get_best_span(
                    start_probs, end_probs
                )

                # Map back to raw indices
                # We need to find the raw index corresponding to the clean index
                # idx_map: raw_idx -> clean_idx
                # We construct reverse map dynamically for the relevant tokens

                raw_span_tokens = doc_tokens[
                    best_cand_struct["start_token"] : best_cand_struct["end_token"]
                ]

                # Find raw start
                raw_start_rel = -1
                for r_idx, token in enumerate(raw_span_tokens):
                    # Check if this raw index maps to our predicted clean start
                    if best_map.get(
                        r_idx
                    ) == pred_start_clean and not self.text_processor.is_html_tag(
                        token
                    ):
                        raw_start_rel = r_idx
                        break

                # Find raw end
                raw_end_rel = -1
                # Search backwards or forwards?
                # We need the token that maps to pred_end_clean
                for r_idx, token in enumerate(raw_span_tokens):
                    if best_map.get(
                        r_idx
                    ) == pred_end_clean and not self.text_processor.is_html_tag(token):
                        raw_end_rel = r_idx
                        # Don't break immediately, in case multiple raw tokens map to same (unlikely with this logic but safe)
                        # Actually with current logic, one raw -> one clean.
                        # But tags map to next valid. We ensure token is not tag.
                        break

                short_ans_str = ""
                if raw_start_rel != -1 and raw_end_rel != -1:
                    global_start = best_cand_struct["start_token"] + raw_start_rel
                    global_end = (
                        best_cand_struct["start_token"] + raw_end_rel + 1
                    )  # +1 for exclusive string format
                    short_ans_str = f"{global_start}:{global_end}"

                results.append(
                    {
                        "example_id": f"{example_id}_long",
                        "PredictionString": long_ans_str,
                    }
                )
                results.append(
                    {
                        "example_id": f"{example_id}_short",
                        "PredictionString": short_ans_str,
                    }
                )

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(Config.SUBMISSION_OUTPUT, index=False)
        print(f"Submission saved to {Config.SUBMISSION_OUTPUT}")


def generate_predictions(sample_size=None):
    pipeline = InferencePipeline()
    pipeline.generate_predictions(sample_size=sample_size)
