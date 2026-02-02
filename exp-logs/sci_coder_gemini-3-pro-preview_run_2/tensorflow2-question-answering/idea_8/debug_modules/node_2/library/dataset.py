import os
import json
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import config
from library.data_utils import text_to_indices, pad_sequence, tokenize


class NQDataset(Dataset):
    def __init__(
        self, split="train", vocab=None, load_cached_data=True, limit_size=None
    ):
        """
        PyTorch Dataset for Natural Questions with Kernel-Pooling logic.

        Args:
            split (str): One of "train", "val", "test".
            vocab (dict): Vocabulary mapping token to index.
            load_cached_data (bool): Whether to load processed data from cache.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.vocab = vocab
        self.limit_size = limit_size

        # Determine paths based on split
        if split == "train":
            self.meta_path = config.TRAIN_META_PATH
            self.cache_path = config.TRAIN_FEATURES_PATH
            self.is_train = True
        elif split == "val":
            self.meta_path = config.VAL_META_PATH
            self.cache_path = config.VAL_FEATURES_PATH
            self.is_train = False
        elif split == "test":
            self.meta_path = config.TEST_META_PATH
            self.cache_path = config.TEST_FEATURES_PATH
            self.is_train = False
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load or process data
        self.data_df = self._load_or_process_data(load_cached_data)

        # Flatten data into (question, candidate) pairs
        self.samples = self._prepare_samples()

    def _load_or_process_data(self, load_cached_data):
        """
        Loads data from Parquet cache or processes from scratch.
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading {self.split} features from {self.cache_path}...")
                df = pd.read_parquet(self.cache_path)
                if self.limit_size:
                    df = df.head(self.limit_size)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from Scratch
        print(f"Processing {self.split} data from raw files...")

        # Load metadata to know which examples to pick
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Cite debug_lesson_7: Enforce string type for IDs to match JSON format
        meta_df = pd.read_csv(self.meta_path, dtype={"example_id": str})
        if self.limit_size:
            meta_df = meta_df.head(self.limit_size)

        # Group by file_path to minimize file opening/closing
        grouped = meta_df.groupby("file_path")

        processed_rows = []

        # Yes/No map
        yn_map = {"NONE": 0, "YES": 1, "NO": 2}

        for file_rel_path, group in grouped:
            file_path = os.path.join(config.INPUT_DIR, file_rel_path)

            # Create a lookup for this file's examples
            # Map example_id -> row in metadata (containing labels)
            examples_in_file = group.set_index("example_id").to_dict("index")

            # Optimization: Track progress to stop early
            total_to_find = len(examples_in_file)
            found_count = 0

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    ex_id = str(entry["example_id"])

                    if ex_id not in examples_in_file:
                        continue

                    meta_row = examples_in_file[ex_id]

                    # 1. Process Text
                    doc_text = entry["document_text"]
                    doc_tokens = tokenize(doc_text)  # List of strings

                    q_text = entry["question_text"]
                    q_indices = text_to_indices(q_text, self.vocab)

                    # 2. Process Candidates
                    # We store candidates as a list of lists of indices
                    candidates_indices = []
                    candidates_tokens_info = (
                        []
                    )  # Store start/end token index in doc for label mapping

                    raw_candidates = entry["long_answer_candidates"]
                    for cand in raw_candidates:
                        start = cand["start_token"]
                        end = cand["end_token"]

                        # Extract token slice
                        # Note: NQ tokens are whitespace split, so indices map directly to list
                        cand_tokens_slice = doc_tokens[start:end]

                        # Convert to indices
                        # We use a helper here to avoid re-tokenizing the slice string
                        unk_idx = self.vocab.get(config.UNK_TOKEN, 1)
                        c_idxs = [self.vocab.get(t, unk_idx) for t in cand_tokens_slice]

                        candidates_indices.append(c_idxs)
                        candidates_tokens_info.append((start, end))

                    # 3. Process Labels (if available)
                    long_answer_index = -1
                    short_answer_spans = (
                        []
                    )  # List of (start_rel, end_rel) relative to candidate
                    yes_no_label = 0

                    if self.is_train or self.split == "val":
                        # Long Answer
                        long_answer_index = int(meta_row.get("long_answer_index", -1))

                        # Short Answer
                        # We need to find the short answer annotation in the raw JSON
                        # because metadata only has boolean 'has_short_answer'
                        # But wait, we are iterating the raw JSON right now.
                        annotations = entry.get("annotations", [])
                        if annotations:
                            ann = annotations[0]

                            # Yes/No
                            yn_str = ann.get("yes_no_answer", "NONE")
                            yes_no_label = yn_map.get(yn_str, 0)

                            # Short Answers
                            # NQ can have multiple short answers, we usually take the first valid one
                            # or all. For training, let's take the first one.
                            shorts = ann.get("short_answers", [])
                            if shorts and long_answer_index != -1:
                                # Short answer is valid only if contained in the long answer
                                # Calculate relative offsets
                                la_cand = raw_candidates[long_answer_index]
                                la_start = la_cand["start_token"]

                                for s in shorts:
                                    s_start = s["start_token"]
                                    s_end = s["end_token"]
                                    # Relative to candidate
                                    rel_start = s_start - la_start
                                    rel_end = s_end - la_start

                                    # Sanity check
                                    if rel_start >= 0 and rel_end <= (
                                        la_cand["end_token"] - la_start
                                    ):
                                        short_answer_spans.append((rel_start, rel_end))
                                        break  # Just take the first valid one

                    # Store row
                    processed_rows.append(
                        {
                            "example_id": ex_id,
                            "q_indices": q_indices,
                            "candidates_indices": candidates_indices,  # List of lists
                            "long_answer_index": long_answer_index,
                            "short_answer_span": (
                                short_answer_spans[0]
                                if short_answer_spans
                                else (-1, -1)
                            ),
                            "yes_no_label": yes_no_label,
                        }
                    )

                    found_count += 1
                    if found_count >= total_to_find:
                        print(
                            f"Found all {total_to_find} requested samples in {file_rel_path}. Stopping read."
                        )
                        break

        # Convert to DataFrame
        df = pd.DataFrame(processed_rows)

        # Save to cache
        # Parquet handles nested lists (arrays) well
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path)
        print(f"Saved {self.split} features to {self.cache_path}. Rows: {len(df)}")

        return df

    def _prepare_samples(self):
        """
        Flattens the dataset into a list of samples (question, single_candidate).
        Implements Random Negative Sampling for training.
        """
        samples = []

        # Iterate through the dataframe rows
        for idx, row in self.data_df.iterrows():
            candidates = row["candidates_indices"]
            num_candidates = len(candidates)
            la_idx = row["long_answer_index"]

            if self.is_train:
                # Training Logic: Positive + Negatives

                # 1. Positive Sample (if exists)
                if la_idx != -1 and la_idx < num_candidates:
                    samples.append(
                        {
                            "row_idx": idx,
                            "cand_idx": la_idx,
                            "label_long": 1.0,
                            "label_short": row["short_answer_span"],
                            "label_yesno": row["yes_no_label"],
                        }
                    )

                    # 2. Negative Samples
                    # Create pool of negative indices
                    neg_pool = [i for i in range(num_candidates) if i != la_idx]

                    # Sample K negatives
                    k = config.NEGATIVE_SAMPLES_RATIO
                    if len(neg_pool) > k:
                        chosen_negs = random.sample(neg_pool, k)
                    else:
                        chosen_negs = neg_pool

                    for neg_idx in chosen_negs:
                        samples.append(
                            {
                                "row_idx": idx,
                                "cand_idx": neg_idx,
                                "label_long": 0.0,
                                "label_short": (
                                    -1,
                                    -1,
                                ),  # No short answer in negative candidate
                                "label_yesno": 0,  # NONE
                            }
                        )
                else:
                    # If no positive answer, we can either skip or sample only negatives.
                    # To expose the model to "unanswerable" questions, we sample some negatives.
                    neg_pool = list(range(num_candidates))
                    k = config.NEGATIVE_SAMPLES_RATIO
                    if len(neg_pool) > k:
                        chosen_negs = random.sample(neg_pool, k)
                    else:
                        chosen_negs = neg_pool

                    for neg_idx in chosen_negs:
                        samples.append(
                            {
                                "row_idx": idx,
                                "cand_idx": neg_idx,
                                "label_long": 0.0,
                                "label_short": (-1, -1),
                                "label_yesno": 0,
                            }
                        )

            else:
                # Validation/Test Logic: All Candidates
                for c_idx in range(num_candidates):
                    # Determine labels if they exist (Validation)
                    is_pos = c_idx == la_idx

                    samples.append(
                        {
                            "row_idx": idx,
                            "cand_idx": c_idx,
                            "label_long": 1.0 if is_pos else 0.0,
                            "label_short": (
                                row["short_answer_span"] if is_pos else (-1, -1)
                            ),
                            "label_yesno": row["yes_no_label"] if is_pos else 0,
                            "example_id": row[
                                "example_id"
                            ],  # Needed for submission grouping
                        }
                    )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        row_idx = sample_info["row_idx"]
        cand_idx = sample_info["cand_idx"]

        # Retrieve data from DataFrame (in memory)
        # Note: accessing by iloc is reasonably fast
        row = self.data_df.iloc[row_idx]

        q_indices = row["q_indices"]
        # Candidates is a list of lists (or numpy array of objects)
        # We need to ensure it's a list
        candidates_list = row["candidates_indices"]
        if isinstance(candidates_list, np.ndarray):
            candidates_list = candidates_list.tolist()

        c_indices = candidates_list[cand_idx]

        # Padding
        q_padded = pad_sequence(q_indices, config.MAX_QUESTION_LENGTH)
        c_padded = pad_sequence(c_indices, config.MAX_CANDIDATE_LENGTH)

        # Labels
        label_long = torch.tensor(sample_info["label_long"], dtype=torch.float32)

        # Short Span Labels
        # We need to clamp indices to max length and handle -1
        s_start, s_end = sample_info["label_short"]

        # If no span (-1), we usually point to 0 (CLS/PAD) or ignore in loss via mask.
        # Here we will use 0 for "no span" but we need to ensure the loss function handles it.
        # Typically for span prediction on negatives, we set target to 0 (if 0 is a special token)
        # or use a specific "ignore_index".
        # In this architecture, we'll clamp to 0.
        if s_start == -1:
            s_start = 0
            s_end = 0
        else:
            # Clamp to max length
            s_start = min(s_start, config.MAX_CANDIDATE_LENGTH - 1)
            s_end = min(s_end, config.MAX_CANDIDATE_LENGTH - 1)

        label_span_start = torch.tensor(s_start, dtype=torch.long)
        label_span_end = torch.tensor(s_end, dtype=torch.long)

        label_yesno = torch.tensor(sample_info["label_yesno"], dtype=torch.long)

        # Return dict
        item = {
            "question": torch.from_numpy(q_padded),
            "candidate": torch.from_numpy(c_padded),
            "label_long": label_long,
            "label_span_start": label_span_start,
            "label_span_end": label_span_end,
            "label_yesno": label_yesno,
        }

        # For inference, pass the example_id to reconstruct predictions
        if not self.is_train and "example_id" in sample_info:
            item["example_id"] = str(sample_info["example_id"])
            # Also pass candidate index to identify which candidate this is
            item["candidate_index"] = cand_idx

        return item
