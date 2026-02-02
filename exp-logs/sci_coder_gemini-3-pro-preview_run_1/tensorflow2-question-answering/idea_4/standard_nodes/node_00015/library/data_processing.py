import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from library.config import Config, set_seed
from library.utils import setup_logger, save_data_cache, load_data_cache

# Initialize logger
logger = setup_logger("data_processing")


class Tokenizer:
    """
    Simple whitespace tokenizer that handles basic preprocessing.
    """

    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.unk_token_id = self.vocab.get(Config.UNK_TOKEN, 1)
        self.pad_token_id = self.vocab.get(Config.PAD_TOKEN, 0)

    def tokenize(self, text):
        """Splits text into tokens."""
        # Basic whitespace tokenization as per dataset description
        return text.split()

    def convert_tokens_to_ids(self, tokens, max_len):
        """Converts tokens to IDs, padding or truncating to max_len."""
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]

        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids += [self.pad_token_id] * (max_len - len(ids))

        return ids


def build_vocab_and_embeddings(train_df, load_cached_data=True):
    """
    Builds vocabulary from training data and initializes embeddings.
    Implements caching mechanism.
    """
    vocab_path = Config.VOCAB_PATH
    emb_path = Config.EMBEDDING_MATRIX_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(vocab_path) and os.path.exists(emb_path):
        logger.info("Loading vocabulary and embeddings from cache...")
        try:
            vocab_df = pd.read_parquet(vocab_path)
            vocab = dict(zip(vocab_df["token"], vocab_df["id"]))
            embedding_matrix = np.load(emb_path, allow_pickle=False)
            return vocab, embedding_matrix
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding...")

    # 2. Build from scratch
    logger.info("Building vocabulary and embeddings from scratch...")

    # Counter for vocabulary
    token_counts = Counter()

    # We need to read the actual text content.
    # To avoid reading the whole massive file, we iterate through the metadata
    # and read samples. For vocab building, we can sample or use the whole train set.
    # Given constraints, we'll use a subset if DEBUG is on, else full.

    data_path = os.path.join(Config.INPUT_DIR, "simplified-nq-train.jsonl")

    # Use a subset for vocab building to save time if needed, but ideally full pass
    # We will iterate over the train_df provided

    with open(data_path, "rb") as f:
        # If dataset is huge, maybe sample 50k rows for vocab to stay within time limits
        sample_indices = train_df.index
        if len(train_df) > 50000:
            sample_indices = random.sample(list(train_df.index), 50000)

        for idx in sample_indices:
            row = train_df.loc[idx]
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Add question tokens
                q_tokens = data.get("question_text", "").split()
                token_counts.update(q_tokens)

                # Add document tokens (maybe just top level or first chunk to save time/memory)
                # Document text can be very long. Let's just take the first 1000 tokens.
                doc_tokens = data.get("document_text", "").split()[:1000]
                token_counts.update(doc_tokens)
            except:
                continue

    # Create vocabulary mapping
    # Start with special tokens
    vocab = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}

    # Add most common tokens up to VOCAB_SIZE
    most_common = token_counts.most_common(Config.VOCAB_SIZE - 2)
    for token, _ in most_common:
        vocab[token] = len(vocab)

    # Initialize Embedding Matrix (Randomly, as we don't have external GloVe file)
    # In a real scenario, we would load GloVe here.
    embedding_matrix = np.random.uniform(
        -0.1, 0.1, (len(vocab), Config.EMBEDDING_DIM)
    ).astype(np.float32)

    # Zero out padding embedding
    embedding_matrix[0] = 0

    # 3. Save to cache
    logger.info("Saving vocabulary and embeddings to cache...")
    vocab_list = [{"token": k, "id": v} for k, v in vocab.items()]
    vocab_df = pd.DataFrame(vocab_list)
    save_data_cache(vocab_df, vocab_path)
    save_data_cache(embedding_matrix, emb_path)

    return vocab, embedding_matrix


def flatten_data(
    metadata_df, data_file_name, is_train=False, load_cached_data=True, cache_path=None
):
    """
    Flattens the hierarchical NQ data into (Question, Candidate) pairs.
    Handles negative sampling for training data.
    """
    if cache_path is None:
        raise ValueError("cache_path must be provided")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading flattened data from {cache_path}...")
        try:
            return load_data_cache(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load flattened data: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info(f"Flattening data from {data_file_name} (is_train={is_train})...")

    records = []
    input_path = os.path.join(Config.INPUT_DIR, data_file_name)

    # Pre-filter metadata if debugging
    if Config.DEBUG:
        metadata_df = metadata_df.head(Config.DEBUG_SIZE)

    with open(input_path, "rb") as f:
        for _, row in metadata_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = entry["question_text"]
            doc_tokens = entry["document_text"].split()
            candidates = entry["long_answer_candidates"]

            # Parse Annotations if training
            correct_long_candidates = set()
            short_answers = []  # List of (start, end) relative to document

            if is_train:
                # We need to parse the annotations string from metadata if available,
                # or use the one in the JSON line. The metadata has it as a string.
                # However, for accuracy, let's use the one in the loaded JSON line
                # since we are already reading it.
                annotations = entry.get("annotations", [])

                for ann in annotations:
                    # Long Answer
                    la = ann.get("long_answer", {})
                    la_start = la.get("start_token", -1)
                    la_end = la.get("end_token", -1)

                    # Find which candidate corresponds to this long answer
                    if la_start != -1:
                        # We need to match this span to a candidate index
                        # But simpler: just store the tokens and match later
                        pass

                    # Short Answers
                    sas = ann.get("short_answers", [])
                    for sa in sas:
                        short_answers.append((sa["start_token"], sa["end_token"]))

                    # Identify correct long answer candidates
                    # A candidate is correct if it matches the long answer annotation
                    # OR if it contains a short answer.
                    # NQ evaluation says: "A long answer... must match exactly the token indices"

                    if la_start != -1:
                        # Find the candidate that matches exactly
                        for i, cand in enumerate(candidates):
                            if (
                                cand["start_token"] == la_start
                                and cand["end_token"] == la_end
                            ):
                                correct_long_candidates.add(i)
                                break

            # Generate Pairs
            # For test set, we keep all candidates (or top level ones).
            # For train set, we do negative sampling.

            # Optimization: Only consider top_level candidates to reduce size
            candidates_to_process = [
                (i, c) for i, c in enumerate(candidates) if c["top_level"]
            ]

            for cand_idx, cand in candidates_to_process:
                c_start = cand["start_token"]
                c_end = cand["end_token"]

                # Extract candidate text
                # Guard against index out of bounds
                c_tokens = doc_tokens[c_start:c_end]
                candidate_text = " ".join(c_tokens)

                # Determine Labels
                is_long_answer = False
                sa_start_idx = -1
                sa_end_idx = -1

                if is_train:
                    if cand_idx in correct_long_candidates:
                        is_long_answer = True

                    # Check for short answer containment
                    # If multiple short answers, pick the first one contained
                    for sa_s, sa_e in short_answers:
                        if sa_s >= c_start and sa_e <= c_end:
                            is_long_answer = True  # Containing short answer implies valid long answer context
                            # Relative indices
                            sa_start_idx = sa_s - c_start
                            sa_end_idx = (
                                sa_e - c_start - 1
                            )  # Inclusive end index for model usually
                            # Ensure indices are within MAX_CANDIDATE_LEN logic handled in Dataset
                            break

                    # Negative Sampling
                    if not is_long_answer:
                        if random.random() > Config.NEGATIVE_SAMPLING_RATE:
                            continue

                records.append(
                    {
                        "example_id": entry["example_id"],
                        "candidate_index": cand_idx,
                        "question_text": question_text,
                        "candidate_text": candidate_text,
                        "label_long": 1 if is_long_answer else 0,
                        "label_short_start": sa_start_idx,
                        "label_short_end": sa_end_idx,
                        "global_start_token": c_start,
                        "global_end_token": c_end,
                    }
                )

    df = pd.DataFrame(records)

    # 3. Save to cache
    logger.info(f"Saving flattened data to {cache_path}...")
    save_data_cache(df, cache_path)

    return df


class NQDataset(Dataset):
    """
    PyTorch Dataset for Natural Questions.
    """

    def __init__(self, dataframe, vocab):
        self.data = dataframe
        self.tokenizer = Tokenizer(vocab)
        self.max_q_len = Config.MAX_QUESTION_LEN
        self.max_c_len = Config.MAX_CANDIDATE_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize
        q_tokens = self.tokenizer.tokenize(row["question_text"])
        c_tokens = self.tokenizer.tokenize(row["candidate_text"])

        # Convert to IDs
        q_ids = self.tokenizer.convert_tokens_to_ids(q_tokens, self.max_q_len)
        c_ids = self.tokenizer.convert_tokens_to_ids(c_tokens, self.max_c_len)

        # Prepare Targets
        label_long = float(row["label_long"])

        # Short answer targets:
        # If no short answer (-1), we set target to ignore_index or a specific class.
        # Here we assume CrossEntropyLoss with ignore_index=-1
        s_start = row["label_short_start"]
        s_end = row["label_short_end"]

        # Adjust for truncation
        if s_start >= self.max_c_len:
            s_start = -1
        if s_end >= self.max_c_len:
            s_end = -1

        # If one is missing/truncated, invalidate both for safety
        if s_start == -1 or s_end == -1:
            s_start = -1
            s_end = -1

        return {
            "q_input": torch.tensor(q_ids, dtype=torch.long),
            "c_input": torch.tensor(c_ids, dtype=torch.long),
            "label_long": torch.tensor(label_long, dtype=torch.float),
            "label_short_start": torch.tensor(s_start, dtype=torch.long),
            "label_short_end": torch.tensor(s_end, dtype=torch.long),
            "example_id": row["example_id"],
            "candidate_index": row["candidate_index"],
            "global_start": row["global_start_token"],
            "global_end": row["global_end_token"],
        }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # 1. Load Metadata
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH)
    val_meta = pd.read_parquet(Config.VAL_META_PATH)
    test_meta = pd.read_parquet(Config.TEST_META_PATH)

    # 2. Build/Load Vocab & Embeddings (using Train data)
    vocab, embedding_matrix = build_vocab_and_embeddings(train_meta, load_cached_data)

    # 3. Flatten Data
    train_df = flatten_data(
        train_meta,
        "simplified-nq-train.jsonl",
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.TRAIN_FLATTENED_PATH,
    )

    val_df = flatten_data(
        val_meta,
        "simplified-nq-train.jsonl",
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.VAL_FLATTENED_PATH,
    )

    test_df = flatten_data(
        test_meta,
        "simplified-nq-test.jsonl",
        is_train=False,
        load_cached_data=load_cached_data,
        cache_path=Config.TEST_FLATTENED_PATH,
    )

    # 4. Create Datasets
    train_dataset = NQDataset(train_df, vocab)
    val_dataset = NQDataset(val_df, vocab)
    test_dataset = NQDataset(test_df, vocab)

    # 5. Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, embedding_matrix
