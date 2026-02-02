import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_metadata(split):
    """
    Loads the metadata parquet file for a specific split (train, val, test).
    """
    if split == "train":
        path = os.path.join(Config.METADATA_DIR, Config.TRAIN_META_FILE)
    elif split == "val":
        path = os.path.join(Config.METADATA_DIR, Config.VAL_META_FILE)
    elif split == "test":
        path = os.path.join(Config.METADATA_DIR, Config.TEST_META_FILE)
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_parquet(path)


def build_vocab(metadata_df, data_file, vocab_size=Config.VOCAB_SIZE, load_cached=True):
    """
    Builds a vocabulary from the training data or loads it if cached.
    """
    vocab_path = Config.VOCAB_SAVE_PATH

    if load_cached and os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, "r") as f:
            return json.load(f)

    print("Building vocabulary from scratch...")
    counter = Counter()

    # Use a sample for vocab building to save time if dataset is huge
    sample_df = metadata_df
    if Config.DEBUG_SAMPLE_SIZE:
        sample_df = metadata_df.head(Config.DEBUG_SAMPLE_SIZE)
    elif len(metadata_df) > 50000:
        sample_df = metadata_df.sample(50000, random_state=Config.SEED)

    file_path = os.path.join(Config.INPUT_DIR, data_file)

    with open(file_path, "rb") as f:
        for _, row in sample_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Add question words
                q_tokens = data.get("question_text", "").split()
                counter.update(q_tokens)

                # Add document words (from a few candidates to get coverage)
                doc_text = data.get("document_text", "")
                doc_tokens = doc_text.split()

                # We can't add all doc tokens for memory reasons, just sample some candidates
                candidates = data.get("long_answer_candidates", [])
                if candidates:
                    # Pick up to 3 candidates randomly
                    for c in random.sample(candidates, min(len(candidates), 3)):
                        start = c["start_token"]
                        end = c["end_token"]
                        if start < len(doc_tokens) and end <= len(doc_tokens):
                            counter.update(doc_tokens[start:end])

            except json.JSONDecodeError:
                continue

    # Create vocab mapping
    most_common = counter.most_common(vocab_size - 2)  # Reserve spots for PAD and UNK

    word2idx = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)

    # Save vocab
    Config.ensure_directories()
    with open(vocab_path, "w") as f:
        json.dump(word2idx, f)

    print(f"Vocabulary built with {len(word2idx)} tokens.")
    return word2idx


def text_to_indices(text_tokens, word2idx, max_len):
    """
    Converts a list of tokens to a list of integers using word2idx.
    Pads or truncates to max_len.
    """
    unk_idx = word2idx[Config.UNK_TOKEN]
    pad_idx = word2idx[Config.PAD_TOKEN]

    # Map tokens to indices
    indices = [word2idx.get(t, unk_idx) for t in text_tokens]

    # Truncate
    if len(indices) > max_len:
        indices = indices[:max_len]

    # Pad
    if len(indices) < max_len:
        indices += [pad_idx] * (max_len - len(indices))

    return indices


def flatten_dataset(metadata_df, data_file, word2idx, is_train=False):
    """
    Reads raw data, flattens hierarchical structure into (Question, Candidate) pairs,
    labels them, and tokenizes text.
    Returns a DataFrame.
    """
    print(f"Flattening dataset from {data_file}...")

    records = []
    file_path = os.path.join(Config.INPUT_DIR, data_file)

    # Debugging limit
    if Config.DEBUG_SAMPLE_SIZE and len(metadata_df) > Config.DEBUG_SAMPLE_SIZE:
        metadata_df = metadata_df.head(Config.DEBUG_SAMPLE_SIZE)

    with open(file_path, "rb") as f:
        for _, row in metadata_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            example_id = entry["example_id"]
            q_text = entry["question_text"]
            doc_text = entry["document_text"]
            doc_tokens = doc_text.split()  # Tokenize by whitespace to match indices
            candidates = entry["long_answer_candidates"]

            # Parse Annotations if training/val
            ground_truth_long = set()  # Set of (start, end) tuples
            ground_truth_short = []  # List of dicts with start, end, yes_no

            if is_train and row["annotations"] is not None:
                try:
                    anns = json.loads(row["annotations"])
                    for ann in anns:
                        # Long answer
                        la = ann.get("long_answer", {})
                        if la.get("start_token", -1) != -1:
                            ground_truth_long.add((la["start_token"], la["end_token"]))

                        # Short answers (can be multiple)
                        sas = ann.get("short_answers", [])
                        yes_no = ann.get("yes_no_answer", "NONE")

                        # If yes/no exists, we treat it as a short answer type
                        if yes_no != "NONE":
                            # Yes/No answers are usually associated with the long answer span
                            # We store a special marker or just the long answer span with label
                            # For simplicity in this architecture, we focus on span extraction.
                            # If yes/no, we might point to the CLS token or similar,
                            # but here we will just mark the containing long answer.
                            pass

                        for sa in sas:
                            ground_truth_short.append(
                                {
                                    "start": sa["start_token"],
                                    "end": sa["end_token"],
                                    "yes_no": yes_no,
                                }
                            )
                except:
                    pass

            # Tokenize Question once per example
            q_indices = text_to_indices(q_text.split(), word2idx, Config.MAX_Q_LEN)

            # Process Candidates
            positives = []
            negatives = []

            for cand in candidates:
                c_start = cand["start_token"]
                c_end = cand["end_token"]

                # Extract candidate text tokens
                # Safe slicing
                c_tokens_text = doc_tokens[c_start:c_end]
                c_indices = text_to_indices(c_tokens_text, word2idx, Config.MAX_C_LEN)

                # Determine Labels
                label_long = 0
                short_start_idx = 0
                short_end_idx = 0

                if is_train:
                    # Check Long Answer Match
                    if (c_start, c_end) in ground_truth_long:
                        label_long = 1

                        # Check for Short Answer containment
                        # We take the first matching short answer span
                        for sa in ground_truth_short:
                            # Check if short answer is strictly contained in candidate
                            if sa["start"] >= c_start and sa["end"] <= c_end:
                                # Calculate relative indices
                                rel_start = sa["start"] - c_start
                                rel_end = sa["end"] - c_start  # Exclusive end index

                                # Check if it fits within the truncated sequence length
                                if rel_end <= Config.MAX_C_LEN:
                                    short_start_idx = rel_start
                                    short_end_idx = rel_end
                                    break  # Found one valid short answer

                record = {
                    "example_id": example_id,
                    "q_indices": q_indices,
                    "c_indices": c_indices,
                    "label_long": label_long,
                    "short_start": short_start_idx,
                    "short_end": short_end_idx,
                    "global_start": c_start,  # For post-processing
                    "global_end": c_end,
                }

                if is_train:
                    if label_long == 1:
                        positives.append(record)
                    else:
                        negatives.append(record)
                else:
                    # For test/inference, keep all
                    records.append(record)

            # Negative Sampling for Training
            if is_train:
                # Keep all positives
                records.extend(positives)

                # Subsample negatives
                num_neg = int(len(negatives) * Config.NEGATIVE_SAMPLING_RATIO)
                if num_neg > 0:
                    records.extend(random.sample(negatives, num_neg))
                elif not positives and negatives:
                    # If no positives (unanswerable), keep a few negatives to learn "no answer"
                    # Keep at least 1 if available
                    records.extend(random.sample(negatives, min(len(negatives), 1)))

    return pd.DataFrame(records)


def process_data(load_cached_data=True):
    """
    Main function to process data.
    1. Loads metadata.
    2. Builds/Loads vocabulary.
    3. Flattens train/val/test data into tokenized pairs.
    4. Caches results to disk.
    """
    set_seed(Config.SEED)
    Config.ensure_directories()

    cache_dir = Config.WORKING_DIR
    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(Config.VOCAB_SAVE_PATH)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        try:
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            with open(Config.VOCAB_SAVE_PATH, "r") as f:
                word2idx = json.load(f)
            return train_df, val_df, test_df, word2idx
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load Metadata
    print("Loading metadata...")
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    # Build Vocabulary (using training metadata)
    word2idx = build_vocab(
        train_meta, Config.TRAIN_DATA_FILE, load_cached=load_cached_data
    )

    # Process Datasets
    train_df = flatten_dataset(
        train_meta, Config.TRAIN_DATA_FILE, word2idx, is_train=True
    )
    val_df = flatten_dataset(
        val_meta, Config.TRAIN_DATA_FILE, word2idx, is_train=True
    )  # Val has annotations
    test_df = flatten_dataset(
        test_meta, Config.TEST_DATA_FILE, word2idx, is_train=False
    )

    # Save to Cache
    print("Saving processed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    print(f"Data processing complete.")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df, word2idx
