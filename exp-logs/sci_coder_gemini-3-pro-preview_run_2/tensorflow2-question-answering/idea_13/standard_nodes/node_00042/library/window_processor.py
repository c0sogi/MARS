import os
import json
import numpy as np
import pandas as pd
from library.config import Config
from library.vocab_manager import VocabManager


class WindowProcessor:
    """
    Handles the segmentation of documents into windows and generation of
    training targets for the Window-Based Max-Pooling Network.
    """

    def __init__(self, config: Config, vocab_manager: VocabManager):
        self.config = config
        self.vocab_manager = vocab_manager

        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

    def _slice_into_windows(self, token_ids):
        """
        Slices a list of token IDs into overlapping windows.

        Args:
            token_ids (list): List of integer token indices.

        Returns:
            list of lists: List of window token sequences.
            list of tuples: List of (start, end) indices relative to the original sequence.
        """
        windows = []
        ranges = []
        seq_len = len(token_ids)

        if seq_len == 0:
            return [], []

        stride = self.config.WINDOW_STRIDE
        win_size = self.config.WINDOW_SIZE

        # Generate windows
        for i in range(0, seq_len, stride):
            # If we are near the end, ensure we don't produce a tiny window
            # unless it's the only one.
            # Strategy: Simple slicing with padding later.

            chunk = token_ids[i : i + win_size]

            # Record the global range of this window
            start_idx = i
            end_idx = i + len(chunk)

            # Pad if necessary (though usually handled by collate_fn,
            # we do it here for uniformity in storage if needed,
            # but storing variable length lists in parquet is fine).
            # We will store raw chunks and pad in the model/dataloader.
            if len(chunk) < win_size:
                padding = [self.vocab_manager.pad_idx] * (win_size - len(chunk))
                chunk = chunk + padding

            windows.append(chunk)
            ranges.append((start_idx, end_idx))

            # Stop if we've covered the whole sequence
            if end_idx == seq_len:
                break

            # Limit number of windows per candidate to save memory/compute
            if len(windows) >= self.config.MAX_WINDOWS_PER_CANDIDATE:
                break

        return windows, ranges

    def process_single_example(self, entry, is_train=True):
        """
        Processes a single NQ example into a list of window features.

        Args:
            entry (dict): Parsed JSON line from NQ dataset.
            is_train (bool): Whether to generate labels.

        Returns:
            list[dict]: List of feature dictionaries for valid windows.
        """
        doc_text = entry.get("document_text", "")
        # Note: NQ document_text is space-separated tokens.
        # We map them to vocab indices.
        doc_indices = self.vocab_manager.text_to_indices(doc_text)

        question_text = entry.get("question_text", "")
        q_indices = self.vocab_manager.text_to_indices(question_text)
        # Truncate question
        q_indices = q_indices[: self.config.MAX_QUESTION_LEN]
        # Pad question
        if len(q_indices) < self.config.MAX_QUESTION_LEN:
            q_indices += [self.vocab_manager.pad_idx] * (
                self.config.MAX_QUESTION_LEN - len(q_indices)
            )

        example_id = entry.get("example_id")
        candidates = entry.get("long_answer_candidates", [])

        # Extract Targets
        target_long_idx = -1
        target_short_start = -1
        target_short_end = -1
        target_yes_no = "NONE"

        if is_train:
            annotations = entry.get("annotations", [])
            if annotations:
                ann = annotations[0]
                target_long_idx = ann.get("long_answer", {}).get("candidate_index", -1)
                target_yes_no = ann.get("yes_no_answer", "NONE")

                shorts = ann.get("short_answers", [])
                if shorts:
                    # Use the first short answer span
                    target_short_start = shorts[0].get("start_token", -1)
                    target_short_end = shorts[0].get("end_token", -1)

        window_features = []

        for cand_idx, cand in enumerate(candidates):
            # Only process top-level candidates to reduce noise/redundancy
            if not cand.get("top_level", False):
                continue

            c_start = cand["start_token"]
            c_end = cand["end_token"]

            # Extract candidate text
            cand_tokens = doc_indices[c_start:c_end]

            # Slice into windows
            windows, ranges = self._slice_into_windows(cand_tokens)

            for w_idx, (w_tokens, (r_start, r_end)) in enumerate(zip(windows, ranges)):
                # Calculate global token indices for this window
                global_w_start = c_start + r_start
                global_w_end = c_start + r_end

                # Default Labels
                label_window = 0
                label_start = 0
                label_end = 0
                label_yes_no = 0  # 0: NONE, 1: YES, 2: NO

                if is_train:
                    # Determine Window Relevance (Binary Classification)
                    is_relevant = False

                    # Case 1: Contains Short Answer
                    if target_short_start != -1:
                        # Check if short answer is fully contained in this window
                        if (target_short_start >= global_w_start) and (
                            target_short_end <= global_w_end
                        ):
                            is_relevant = True
                            # Calculate relative span indices
                            # Note: w_tokens might be padded, but r_start is relative to candidate start
                            label_start = target_short_start - global_w_start
                            # Convert exclusive end index to inclusive for classification target
                            label_end = target_short_end - global_w_start - 1

                    # Case 2: Yes/No Question (Target Long Answer matches this candidate)
                    elif target_yes_no != "NONE" and target_long_idx == cand_idx:
                        # If it's the correct long answer, we mark windows as relevant
                        # to train the Yes/No head.
                        is_relevant = True

                    # Case 3: Long Answer Only (No short, No Yes/No)
                    elif target_long_idx == cand_idx:
                        # We mark it relevant for ranking purposes
                        is_relevant = True

                    if is_relevant:
                        label_window = 1

                    # Map Yes/No string to int
                    if target_yes_no == "YES":
                        label_yes_no = 1
                    elif target_yes_no == "NO":
                        label_yes_no = 2
                    else:
                        label_yes_no = 0

                # Construct feature dict
                feat = {
                    "example_id": str(example_id),
                    "candidate_index": cand_idx,
                    "window_index": w_idx,
                    "input_ids": w_tokens,  # List[int]
                    "question_ids": q_indices,  # List[int]
                    "label_window": label_window,
                    "label_start": label_start,
                    "label_end": label_end,
                    "label_yes_no": label_yes_no,
                    # Metadata for inference reconstruction
                    "global_start": global_w_start,
                    "global_end": global_w_end,
                }
                window_features.append(feat)

        return window_features

    def process_dataset(self, load_cached_data=True, is_train=True):
        """
        Processes the entire dataset defined in Config.
        Checks for cached parquet file first.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            is_train (bool): True for training data, False for test data.

        Returns:
            pd.DataFrame: DataFrame containing processed window features.
        """
        split_name = "train" if is_train else "test"
        cache_filename = f"{split_name}_features.parquet"
        cache_path = os.path.join(self.config.CACHE_DIR, cache_filename)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                print(f"Loaded {len(df)} windows.")
                return df
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute
        source_path = (
            self.config.TRAIN_DATA_PATH if is_train else self.config.TEST_DATA_PATH
        )
        print(f"Processing {source_path}...")

        # Use metadata to filter/select if needed, but here we iterate the file directly
        # to ensure we get the text content.
        # However, for training, we might want to limit to the samples in train_metadata.csv
        # if we were doing a strict split here.
        # Given the prompt instructions, we process the file provided in config.

        features_list = []
        count = 0
        limit = self.config.DEBUG_SIZE if self.config.DEBUG else None

        # Pre-load metadata IDs to filter if we want strictly what's in metadata
        # (Optional optimization, skipping for simplicity as we process the whole file)

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit and count >= limit:
                        break

                    entry = json.loads(line)
                    feats = self.process_single_example(entry, is_train=is_train)
                    features_list.extend(feats)

                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} documents...")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {source_path}")

        df = pd.DataFrame(features_list)

        # 3. Save Cache
        print(f"Saving {len(df)} features to {cache_path}...")
        # Ensure lists are stored correctly in parquet (pyarrow handles lists)
        df.to_parquet(cache_path, index=False)

        return df
