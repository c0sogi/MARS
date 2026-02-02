import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.data_utils import Tokenizer, ensure_dir


class NQDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        raw_data_path,
        tokenizer=None,
        is_train=True,
        load_cached_data=True,
        debug=Config.DEBUG,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    ):
        """
        Dataset class for Natural Questions.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            raw_data_path (str): Path to the raw JSONL file.
            tokenizer (Tokenizer): Instance of Tokenizer class.
            is_train (bool): Whether this is a training dataset (enables negative sampling).
            load_cached_data (bool): Whether to load processed features from cache.
            debug (bool): If True, limits the dataset size.
            sample_size (int): Number of samples to use in debug mode.
        """
        self.metadata_path = metadata_path
        self.raw_data_path = raw_data_path
        self.is_train = is_train
        self.debug = debug
        self.sample_size = sample_size

        # Initialize or load tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer()
            if os.path.exists(Config.VOCAB_CACHE_FILE):
                self.tokenizer.load(Config.VOCAB_CACHE_FILE)
            else:
                # If no tokenizer provided and no cache, we can't process text properly
                # In a real pipeline, we'd fit it here, but for this module we assume
                # vocabulary is available or passed in.
                print("Warning: No tokenizer provided and no vocab cache found.")
        else:
            self.tokenizer = tokenizer

        # Determine cache file path based on split
        if "train" in metadata_path and is_train:
            self.cache_file = Config.TRAIN_FEATURES_CACHE
        elif "validation" in metadata_path:
            self.cache_file = Config.VAL_FEATURES_CACHE
        elif "test" in metadata_path:
            self.cache_file = Config.TEST_FEATURES_CACHE
        else:
            # Fallback for custom splits
            base_name = os.path.basename(metadata_path).replace(".csv", "")
            self.cache_file = os.path.join(
                Config.WORKING_DIR, f"{base_name}_features.parquet"
            )

        # Load and process data
        self.data = self._process_and_cache(load_cached_data)

    def _process_and_cache(self, load_cached_data):
        """
        Loads data from cache or processes raw JSONL to extract features.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading cached features from {self.cache_file}...")
            try:
                df = pd.read_parquet(self.cache_file)
                if self.debug:
                    df = df.head(self.sample_size)
                print(f"Loaded {len(df)} samples from cache.")
                return df
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing raw data from {self.raw_data_path}...")

        # Load metadata to filter/identify split
        meta_df = pd.read_csv(self.metadata_path)
        valid_ids = set(meta_df["example_id"].astype(str))

        # For training, we might need label info from metadata to speed up,
        # but raw JSONL has annotations. We rely on raw JSONL for content.

        features = []

        # Read JSONL
        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                eid = str(entry["example_id"])

                if eid not in valid_ids:
                    continue

                # Extract basic info
                question_text = entry["question_text"]
                doc_text = entry["document_text"]
                doc_tokens = (
                    doc_text.split()
                )  # Whitespace tokenization as per description
                candidates = entry["long_answer_candidates"]

                # Determine targets
                long_answer_idx = -1
                short_answers = []
                yes_no = "NONE"

                if self.is_train or "validation" in self.metadata_path:
                    # Extract ground truth from annotations
                    if len(entry["annotations"]) > 0:
                        ann = entry["annotations"][0]
                        long_answer_idx = ann["long_answer"]["candidate_index"]
                        short_answers = ann[
                            "short_answers"
                        ]  # List of dicts {start_token, end_token}
                        yes_no = ann["yes_no_answer"]

                # --- Candidate Selection Logic ---
                selected_candidates = []

                if self.is_train:
                    # Training: Random Negative Sampling
                    # 1. Positive sample (if exists)
                    if long_answer_idx != -1:
                        selected_candidates.append(
                            {"cand_idx": long_answer_idx, "label": 1, "is_gold": True}
                        )

                    # 2. Negative samples
                    # Identify all indices that are NOT the long answer
                    all_indices = list(range(len(candidates)))
                    neg_indices = [i for i in all_indices if i != long_answer_idx]

                    # Determine how many negatives to sample
                    # If we have a positive, we take NEGATIVE_RATIO * 1
                    # If no positive, we might still take negatives (background)
                    num_neg = int(Config.NEGATIVE_RATIO)
                    if num_neg > 0 and len(neg_indices) > 0:
                        sampled_negs = random.sample(
                            neg_indices, min(len(neg_indices), num_neg)
                        )
                        for idx in sampled_negs:
                            selected_candidates.append(
                                {"cand_idx": idx, "label": 0, "is_gold": False}
                            )
                else:
                    # Validation/Test: Use ALL candidates
                    for i in range(len(candidates)):
                        # Label is 1 if it matches ground truth, else 0
                        is_match = i == long_answer_idx
                        selected_candidates.append(
                            {
                                "cand_idx": i,
                                "label": 1 if is_match else 0,
                                "is_gold": is_match,
                            }
                        )

                # --- Feature Extraction for Selected Candidates ---
                q_seq = self.tokenizer.texts_to_sequences([question_text])[0]

                for item in selected_candidates:
                    c_idx = item["cand_idx"]
                    cand_struct = candidates[c_idx]
                    start_token = cand_struct["start_token"]
                    end_token = cand_struct["end_token"]

                    # Extract text for this candidate
                    # Handle bounds safely
                    cand_tokens_list = doc_tokens[start_token:end_token]
                    cand_text = " ".join(cand_tokens_list)
                    c_seq = self.tokenizer.texts_to_sequences([cand_text])[0]

                    # Short Answer Targets (Relative to Candidate)
                    s_start = -1
                    s_end = -1

                    if item["is_gold"] and short_answers:
                        # We have short answers, find the first one that fits in this candidate
                        # NQ annotations are document-relative.
                        # We need to map them to candidate-relative.
                        for sa in short_answers:
                            sa_start = sa["start_token"]
                            sa_end = sa["end_token"]

                            # Check if short answer is strictly within candidate
                            if sa_start >= start_token and sa_end <= end_token:
                                # Calculate relative offset
                                # Note: tokenizer splits by whitespace, which aligns with doc_tokens
                                s_start = sa_start - start_token
                                s_end = (
                                    sa_end - start_token - 1
                                )  # Inclusive end index for span
                                break

                    # Yes/No Target
                    yn_label = 0  # NONE
                    if item["is_gold"]:
                        if yes_no == "YES":
                            yn_label = 1
                        elif yes_no == "NO":
                            yn_label = 2

                    features.append(
                        {
                            "example_id": eid,
                            "candidate_index": c_idx,
                            "q_seq": q_seq,
                            "c_seq": c_seq,
                            "long_label": item["label"],
                            "short_start": s_start,
                            "short_end": s_end,
                            "yes_no_label": yn_label,
                        }
                    )

                if self.debug and len(features) >= self.sample_size:
                    break

        # Create DataFrame
        df = pd.DataFrame(features)

        # Save to cache
        ensure_dir(self.cache_file)
        # Parquet doesn't support lists natively well in all engines, but pandas pyarrow handles it.
        # Ensure directory exists
        df.to_parquet(self.cache_file, index=False)
        print(f"Saved processed features to {self.cache_file}")

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Pad Question
        q_seq = row["q_seq"]
        if isinstance(q_seq, np.ndarray):
            q_seq = q_seq.tolist()

        if len(q_seq) > Config.MAX_Q_LEN:
            q_seq = q_seq[: Config.MAX_Q_LEN]
        else:
            q_seq = q_seq + [0] * (Config.MAX_Q_LEN - len(q_seq))

        # 2. Pad Candidate
        c_seq = row["c_seq"]
        if isinstance(c_seq, np.ndarray):
            c_seq = c_seq.tolist()

        original_c_len = len(c_seq)
        if len(c_seq) > Config.MAX_DOC_LEN:
            c_seq = c_seq[: Config.MAX_DOC_LEN]
        else:
            c_seq = c_seq + [0] * (Config.MAX_DOC_LEN - len(c_seq))

        # 3. Adjust Targets
        long_label = row["long_label"]
        yes_no_label = row["yes_no_label"]

        s_start = row["short_start"]
        s_end = row["short_end"]

        # If short answer is truncated by padding, invalidate it
        if s_start >= Config.MAX_DOC_LEN or s_end >= Config.MAX_DOC_LEN:
            s_start = -1
            s_end = -1

        # Convert to Tensors
        return {
            "q_seq": torch.tensor(q_seq, dtype=torch.long),
            "c_seq": torch.tensor(c_seq, dtype=torch.long),
            "long_label": torch.tensor(
                long_label, dtype=torch.float
            ),  # BCE expects float
            "short_start": torch.tensor(
                s_start, dtype=torch.long
            ),  # CrossEntropy expects long
            "short_end": torch.tensor(s_end, dtype=torch.long),
            "yes_no_label": torch.tensor(yes_no_label, dtype=torch.long),
            "example_id": row["example_id"],
            "candidate_index": row["candidate_index"],
        }
