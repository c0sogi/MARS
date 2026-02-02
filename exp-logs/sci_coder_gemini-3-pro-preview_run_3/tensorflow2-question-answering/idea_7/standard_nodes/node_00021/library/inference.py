import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import re
from library.utils import (
    ensure_dir,
    load_embeddings,
    load_jsonl_sample,
    CACHE_DIR,
    INPUT_DIR,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_EMBEDDING_DIM,
    get_dataset_partitions,
)
from library.models import CompareAggregateRanker, DilatedConvReader
from library.data_loader import get_tokenizer

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class InferencePipeline:
    def __init__(
        self, cache_dir=CACHE_DIR, submission_dir="./working/demo_execution/submission"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.submission_dir = submission_dir
        ensure_dir(self.cache_dir)
        ensure_dir(self.submission_dir)

        self.tokenizer = None
        self.ranker = None
        self.reader = None
        self.embedding_matrix = None

    def load_resources(self, load_cached_data=True):
        """
        Loads tokenizer and models.
        """
        print("Loading resources for inference...")

        # 1. Load Metadata to initialize tokenizer (if not cached)
        train_meta, _, test_meta = get_dataset_partitions()

        # 2. Tokenizer
        self.tokenizer = get_tokenizer(train_meta, load_cached_data=load_cached_data)

        # 3. Embeddings
        self.embedding_matrix = load_embeddings(
            self.tokenizer.word_index,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            load_cached_data=load_cached_data,
        )

        # 4. Initialize Models
        self.ranker = CompareAggregateRanker(self.embedding_matrix).to(self.device)
        self.reader = DilatedConvReader(self.embedding_matrix).to(self.device)

        # 5. Load Weights
        ranker_path = os.path.join(self.cache_dir, "ranker_best.pth")
        reader_path = os.path.join(self.cache_dir, "reader_best.pth")

        if os.path.exists(ranker_path):
            self.ranker.load_state_dict(
                torch.load(ranker_path, map_location=self.device)
            )
            print(f"Ranker weights loaded from {ranker_path}")
        else:
            print(
                f"Warning: Ranker weights not found at {ranker_path}. Using initialized weights."
            )

        if os.path.exists(reader_path):
            self.reader.load_state_dict(
                torch.load(reader_path, map_location=self.device)
            )
            print(f"Reader weights loaded from {reader_path}")
        else:
            print(
                f"Warning: Reader weights not found at {reader_path}. Using initialized weights."
            )

        self.ranker.eval()
        self.reader.eval()

    def preprocess_test_data(self, load_cached_data=True, sample_size=None):
        """
        Reads test metadata and processes each document to extract candidates and tokens.
        Returns a list of dictionaries (one per example).
        """
        cache_file = os.path.join(self.cache_dir, "ranker_test_features.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading preprocessed test data from {cache_file}")
            # Parquet doesn't handle nested lists of arbitrary depth well in all pandas versions without pyarrow tweaks,
            # but we will try to read it. If complex types were saved, they should load.
            try:
                df = pd.read_parquet(cache_file)
                # Convert DataFrame back to list of dicts for easier iteration
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load cached test data: {e}. Recomputing.")

        print("Preprocessing test data...")
        _, _, test_meta = get_dataset_partitions()

        if sample_size and len(test_meta) > sample_size:
            test_meta = test_meta.iloc[:sample_size]

        processed_data = []

        for _, row in test_meta.iterrows():
            file_path = os.path.join(INPUT_DIR, row["file_path"])
            data = load_jsonl_sample(file_path, row["byte_offset"])
            if not data:
                continue

            example_id = data["example_id"]
            question_text = data["question_text"]
            document_text = data["document_text"]
            doc_tokens = document_text.split()

            # Tokenize Question
            q_seq = self.tokenizer.texts_to_sequences([question_text])[0]

            # Extract Candidates
            candidates = data["long_answer_candidates"]
            cand_seqs = []
            cand_spans = []  # (start_token, end_token)

            for cand in candidates:
                if cand["top_level"]:
                    start = cand["start_token"]
                    end = cand["end_token"]
                    if start < len(doc_tokens):
                        # Extract text and tokenize
                        cand_text = " ".join(doc_tokens[start:end])
                        c_seq = self.tokenizer.texts_to_sequences([cand_text])[0]

                        cand_seqs.append(c_seq)
                        cand_spans.append(
                            [start, end]
                        )  # Use list for parquet compatibility

            if cand_seqs:
                processed_data.append(
                    {
                        "example_id": str(example_id),
                        "q_ids": q_seq,
                        "candidate_ids": cand_seqs,
                        "candidate_spans": cand_spans,
                    }
                )

        # Save to cache
        if processed_data:
            df = pd.DataFrame(processed_data)
            try:
                df.to_parquet(cache_file, index=False)
                print(f"Saved processed test data to {cache_file}")
            except Exception as e:
                print(f"Could not save parquet cache (likely due to nested lists): {e}")

        return processed_data

    def predict(self, test_data, ranker_threshold=0.0, reader_threshold=0.0):
        """
        Runs inference on the processed test data.
        """
        print(f"Running inference on {len(test_data)} examples...")

        results = []

        for i, example in enumerate(test_data):
            example_id = str(example["example_id"])
            # Cite debug_lesson_4: Sanitize Untrusted Identifiers Before Serialization
            # Remove commas, newlines, and quotes to prevent CSV injection
            example_id = re.sub(r'[\n\r,"]', "", example_id)

            # Ensure inputs are python lists to avoid numpy ambiguity and concatenation issues
            q_ids = list(example["q_ids"])
            cand_ids = [list(c) for c in example["candidate_ids"]]
            cand_spans = [list(s) for s in example["candidate_spans"]]

            if len(cand_ids) == 0:
                results.append((example_id + "_long", ""))
                results.append((example_id + "_short", ""))
                continue

            # --- 1. Ranking ---
            # Prepare batch for ranker: (num_candidates, seq_len)
            num_cands = len(cand_ids)

            # Pad Question
            q_tensor = torch.tensor([q_ids] * num_cands, dtype=torch.long).to(
                self.device
            )
            # Pad Candidates
            max_cand_len = max(len(c) for c in cand_ids)
            max_cand_len = min(max_cand_len, DEFAULT_MAX_SEQ_LEN)

            p_tensor = torch.zeros((num_cands, max_cand_len), dtype=torch.long).to(
                self.device
            )
            for j, c in enumerate(cand_ids):
                l = min(len(c), max_cand_len)
                p_tensor[j, :l] = torch.tensor(c[:l], dtype=torch.long)

            with torch.no_grad():
                ranker_logits = self.ranker(q_tensor, p_tensor).squeeze(
                    1
                )  # (num_cands,)
                ranker_probs = torch.sigmoid(ranker_logits)

            best_cand_idx = torch.argmax(ranker_probs).item()
            best_cand_score = ranker_probs[best_cand_idx].item()

            # --- 2. Reading ---
            # Prepare input for reader: Q + Best Paragraph
            best_p_ids = cand_ids[best_cand_idx]
            input_seq = q_ids + best_p_ids

            # Truncate if necessary
            if len(input_seq) > DEFAULT_MAX_SEQ_LEN:
                input_seq = input_seq[:DEFAULT_MAX_SEQ_LEN]

            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)

            with torch.no_grad():
                start_logits, end_logits = self.reader(input_tensor)
                # Softmax to get probs
                start_probs = torch.softmax(start_logits, dim=1).squeeze(0)
                end_probs = torch.softmax(end_logits, dim=1).squeeze(0)

            # --- 3. Span Selection ---
            # Find best valid span (start <= end)
            # We only look for answers inside the paragraph part, not the question part
            p_start_offset = len(q_ids)
            p_end_offset = len(input_seq)

            best_span_score = -1.0
            best_span = (0, 0)

            # Heuristic search: Look for top-k start and end indices to avoid O(N^2)
            # Limit search space to paragraph
            valid_start_probs = start_probs[p_start_offset:p_end_offset]
            valid_end_probs = end_probs[p_start_offset:p_end_offset]

            if len(valid_start_probs) > 0:
                # Get top 5 start and end indices relative to paragraph start
                top_starts = torch.topk(
                    valid_start_probs, k=min(5, len(valid_start_probs))
                ).indices
                top_ends = torch.topk(
                    valid_end_probs, k=min(5, len(valid_end_probs))
                ).indices

                for s_rel in top_starts:
                    for e_rel in top_ends:
                        if s_rel <= e_rel:
                            # Score is product of probabilities (or sum of log probs)
                            score = (
                                valid_start_probs[s_rel].item()
                                * valid_end_probs[e_rel].item()
                            )
                            if score > best_span_score:
                                best_span_score = score
                                best_span = (s_rel.item(), e_rel.item())

            # --- 4. Format Output ---

            # Long Answer Logic
            long_ans_str = ""
            if best_cand_score > ranker_threshold:
                la_start, la_end = cand_spans[best_cand_idx]
                long_ans_str = f"{la_start}:{la_end}"

            # Short Answer Logic
            short_ans_str = ""
            # Short answer is valid only if long answer is valid AND short answer score is high enough
            if long_ans_str and best_span_score > reader_threshold:
                rel_s, rel_e = best_span
                # Map relative indices back to document token indices
                # doc_index = candidate_start + relative_index
                la_start, _ = cand_spans[best_cand_idx]
                sa_doc_start = la_start + rel_s
                # End index in prediction is usually inclusive, but dataset is exclusive.
                # The Reader predicts inclusive end index.
                # Submission format usually expects start:end (exclusive) or token span.
                # NQ format: start:end (exclusive).
                # Reader predicted `rel_e` is inclusive index of the last token.
                # So doc end token index = la_start + rel_e.
                # Exclusive end for string = doc end token index + 1.
                sa_doc_end = la_start + rel_e + 1
                short_ans_str = f"{sa_doc_start}:{sa_doc_end}"

            results.append((example_id + "_long", long_ans_str))
            results.append((example_id + "_short", short_ans_str))

        return results

    def generate_submission(
        self,
        sample_size=None,
        ranker_threshold=0.5,
        reader_threshold=0.1,
        load_cached_data=True,
    ):
        """
        Full inference process generating the submission file.
        """
        # 1. Load resources
        self.load_resources(load_cached_data=load_cached_data)

        # 2. Process Data
        test_data = self.preprocess_test_data(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # 3. Predict
        predictions = self.predict(
            test_data,
            ranker_threshold=ranker_threshold,
            reader_threshold=reader_threshold,
        )

        # 4. Save
        df = pd.DataFrame(predictions, columns=["example_id", "PredictionString"])
        out_path = os.path.join(self.submission_dir, "submission.csv")
        df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
