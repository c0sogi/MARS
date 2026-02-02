import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.preprocessing import TextPreprocessor
from library.data_loader import NQRankerTestDataset, collate_fn
from library.ranker_model import DecomposableAttentionRanker
from library.reader_model import GatedConvReader


class EvalEngine:
    def __init__(self):
        self.device = Config.DEVICE
        self.preprocessor = TextPreprocessor()

        # Load resources
        self.vocab = self.preprocessor.build_vocabulary(load_cached_data=True)
        self.embedding_matrix = self.preprocessor.load_embeddings(load_cached_data=True)

        # Initialize Models
        self.ranker = DecomposableAttentionRanker(self.embedding_matrix).to(self.device)
        self.reader = GatedConvReader(self.embedding_matrix).to(self.device)

        # Load Checkpoints
        self._load_checkpoint(self.ranker, Config.RANKER_MODEL_PATH)
        self._load_checkpoint(self.reader, Config.READER_MODEL_PATH)

        self.ranker.eval()
        self.reader.eval()

    def _load_checkpoint(self, model, path):
        if os.path.exists(path):
            print(f"Loading model checkpoint from {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(
                f"Warning: Checkpoint not found at {path}. Using random initialization."
            )

    def predict_sample(self, load_cached_data=True):
        """
        Runs the full inference pipeline: Ranking -> Reading -> Submission Generation.
        """
        # 1. Load Test Data
        test_dataset = NQRankerTestDataset(
            metadata_path=Config.TEST_METADATA,
            raw_file=Config.TEST_FILE,
            preprocessor=self.preprocessor,
            load_cached_data=load_cached_data,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=os.cpu_count() or 1,
        )

        # 2. Rank Candidates
        print("Running Ranker on test set...")
        best_candidates = self._rank_candidates(test_loader)

        # 3. Read Answers from Best Candidates
        print("Running Reader on selected candidates...")
        predictions = self._read_answers(best_candidates, test_dataset)

        # 4. Generate Submission File
        print("Generating submission file...")
        self._generate_csv(predictions)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    def _rank_candidates(self, dataloader):
        """
        Scores all candidates and selects the best one per example_id.
        Returns a dict: example_id -> (best_candidate_index, best_score)
        """
        results = []

        with torch.no_grad():
            for batch in dataloader:
                q_indices = batch["q_indices"].to(self.device)
                doc_indices = batch["doc_indices"].to(self.device)
                example_ids = batch["example_ids"]
                cand_indices = batch["candidate_indices"]

                logits = self.ranker(q_indices, doc_indices)
                scores = torch.sigmoid(logits).cpu().numpy().flatten()

                for eid, c_idx, score in zip(example_ids, cand_indices, scores):
                    results.append(
                        {"example_id": eid, "candidate_index": c_idx, "score": score}
                    )

        df = pd.DataFrame(results)

        # Find best candidate for each example
        # idxmax returns the index of the max value
        best_indices = df.groupby("example_id")["score"].idxmax()
        best_rows = df.loc[best_indices]

        # Convert to dict for fast lookup
        best_candidates = {}
        for _, row in best_rows.iterrows():
            best_candidates[str(row["example_id"])] = (
                int(row["candidate_index"]),
                float(row["score"]),
            )

        return best_candidates

    def _read_answers(self, best_candidates, dataset):
        """
        Runs the reader on the chosen candidates.
        Reconstructs the input sequence from the dataset items.
        """
        predictions = {}

        # We need to iterate the dataset again to get the text/indices for the best candidates.
        # Since dataset is a list of dicts in memory (after loading), this is fast.

        # Create a mapping for quick access or iterate and filter
        # Iterating is safer to avoid memory spikes if we tried to map everything

        # To batch this efficiently, we collect inputs
        batch_inputs = []
        batch_meta = []

        for item in dataset.data:
            eid = str(item["example_id"])
            c_idx = item["candidate_index"]

            if eid in best_candidates and best_candidates[eid][0] == c_idx:
                ranker_score = best_candidates[eid][1]

                # Reconstruct Reader Input: Q + Candidate
                # Remove padding from stored indices
                q_inds = [idx for idx in item["q_indices"] if idx != 0]
                doc_inds = [idx for idx in item["doc_indices"] if idx != 0]

                input_indices = q_inds + doc_inds
                # Truncate if necessary (though preprocessing should have handled max lens separately)
                max_len = Config.MAX_Q_LEN + Config.MAX_DOC_LEN
                if len(input_indices) > max_len:
                    input_indices = input_indices[:max_len]

                batch_inputs.append(torch.tensor(input_indices, dtype=torch.long))
                batch_meta.append(
                    {
                        "example_id": eid,
                        "q_len": len(q_inds),
                        "ranker_score": ranker_score,
                        "candidate_index": c_idx,
                    }
                )

                # Process in batches
                if len(batch_inputs) >= Config.BATCH_SIZE:
                    self._process_reader_batch(batch_inputs, batch_meta, predictions)
                    batch_inputs = []
                    batch_meta = []

        # Process remaining
        if batch_inputs:
            self._process_reader_batch(batch_inputs, batch_meta, predictions)

        return predictions

    def _process_reader_batch(self, inputs, meta, predictions):
        # Pad inputs
        input_tensor = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=0
        ).to(self.device)

        with torch.no_grad():
            start_logits, end_logits = self.reader(input_tensor)

            # Softmax
            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)

            start_probs = start_probs.cpu().numpy()
            end_probs = end_probs.cpu().numpy()

        for i, m in enumerate(meta):
            q_len = m["q_len"]
            r_score = m["ranker_score"]
            s_prob = start_probs[i]
            e_prob = end_probs[i]

            # Find best span
            best_span_score = -1.0
            best_start = -1
            best_end = -1

            # Search constraints
            # Start must be after question (>= q_len)
            # End must be >= Start
            # Length must be <= MAX_ANSWER_LEN

            # Optimization: Only look at top K start/end tokens to reduce complexity if needed,
            # but full search over short sequence is fine.
            seq_len = len(inputs[i])

            for s in range(q_len, seq_len):
                for e in range(s, min(s + Config.MAX_ANSWER_LEN, seq_len)):
                    score = s_prob[s] * e_prob[e]
                    if score > best_span_score:
                        best_span_score = score
                        best_start = s
                        best_end = e

            # Joint Confidence
            joint_confidence = r_score * best_span_score

            # Convert to local offset within the paragraph
            local_start = best_start - q_len
            local_end = best_end - q_len

            predictions[m["example_id"]] = {
                "candidate_index": m["candidate_index"],
                "local_start_token": local_start,
                "local_end_token": local_end,  # inclusive
                "confidence": joint_confidence,
            }

    def _generate_csv(self, predictions):
        """
        Reads the raw test file to map local predictions to global token offsets
        and writes the submission CSV.
        """
        submission_rows = []

        # We need to read the raw file to get global offsets
        with open(Config.TEST_FILE, "rb") as f:
            # We iterate line by line. Since the test file is not huge (1.7GB),
            # reading it sequentially is acceptable.
            for line in f:
                try:
                    entry = json.loads(line)
                    eid = str(entry["example_id"])

                    pred = predictions.get(eid)

                    long_ans_str = ""
                    short_ans_str = ""

                    if pred and pred["confidence"] >= Config.CONFIDENCE_THRESHOLD:
                        c_idx = pred["candidate_index"]
                        candidates = entry.get("long_answer_candidates", [])

                        # Find the candidate object
                        # Note: The candidate_index in our dataset corresponds to the index
                        # in the filtered list of top-level candidates?
                        # Or the original list?
                        # In preprocessing.py, we iterate `for cand in candidates` and check `top_level`.
                        # The `idx` stored in NQRankerTestDataset is the index in the *original* list?
                        # Let's check preprocessing.py:
                        # `for idx, cand in enumerate(candidates):` -> idx is original index.
                        # Yes.

                        if c_idx < len(candidates):
                            cand = candidates[c_idx]
                            global_start = cand["start_token"]
                            global_end = cand["end_token"]

                            # Long Answer Prediction
                            long_ans_str = f"{global_start}:{global_end}"

                            # Short Answer Prediction
                            # Map local (relative to paragraph start) to global
                            l_start = pred["local_start_token"]
                            l_end = pred["local_end_token"]

                            if l_start >= 0 and l_end >= 0:
                                s_global_start = global_start + l_start
                                s_global_end = (
                                    global_start + l_end + 1
                                )  # +1 for exclusive end in output format usually?
                                # Task description: "start:end token indices".
                                # Usually NQ format is inclusive start, exclusive end for Python slicing,
                                # but the CSV example shows "start:end".
                                # Looking at sample_submission: "105:200".
                                # The annotations in train are `start_token`, `end_token` (exclusive).
                                # We will stick to the standard `start:end` meaning `start` inclusive, `end` exclusive.
                                # Our `local_end` from loop was inclusive. So +1.
                                short_ans_str = f"{s_global_start}:{s_global_end}"

                    # Add rows
                    submission_rows.append([f"{eid}_long", long_ans_str])
                    submission_rows.append([f"{eid}_short", short_ans_str])

                except json.JSONDecodeError:
                    continue

        # Create DataFrame and save
        df = pd.DataFrame(submission_rows, columns=["example_id", "PredictionString"])
        df.to_csv(Config.SUBMISSION_FILE, index=False)
