import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.configuration import Config
from library.text_utils import tokenize, text_to_indices, build_vocab, strip_html_tags
from library.feature_engineering import create_ranker_dataset


class RankerDatasetBuilder:
    """
    Handles the creation and loading of tabular datasets for the Gradient Boosting Ranker.
    Wraps the feature engineering logic to produce data suitable for LightGBM.
    """

    @staticmethod
    def build_train_set(load_cached_data=True, sample_size=None):
        """
        Builds or loads the training dataset for the ranker.

        Args:
            load_cached_data (bool): Whether to try loading from disk first.
            sample_size (int, optional): Number of metadata rows to process (for debugging).

        Returns:
            pd.DataFrame: Tabular dataset with features and labels.
        """
        return create_ranker_dataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            output_path=Config.RANKER_TRAIN_CACHE,
            is_train=True,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )

    @staticmethod
    def build_val_set(load_cached_data=True, sample_size=None):
        """
        Builds or loads the validation dataset for the ranker.
        """
        return create_ranker_dataset(
            metadata_path=Config.VAL_METADATA_PATH,
            output_path=Config.RANKER_VAL_CACHE,
            is_train=True,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )

    @staticmethod
    def build_test_set(load_cached_data=False, sample_size=None):
        """
        Builds the test dataset for the ranker (inference).
        Usually generated on the fly or cached temporarily.
        """
        test_cache_path = os.path.join(
            Config.WORKING_DIR, "ranker_test_features.parquet"
        )
        return create_ranker_dataset(
            metadata_path=Config.TEST_METADATA_PATH,
            output_path=test_cache_path,
            is_train=False,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )


def _process_reader_data(metadata_path, output_path, is_train=True, sample_size=None):
    """
    Internal function to process raw JSONL data into a format suitable for the Reader model.
    Extracts (Question, Context, Start_Token, End_Token) tuples for valid short answers.
    """
    print(f"Generating reader data from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # For training the reader, we only want examples that actually have a short answer
    if is_train:
        if "has_short_answer" in df_meta.columns:
            df_meta = df_meta[df_meta["has_short_answer"] == True]
        else:
            print(
                "Warning: 'has_short_answer' column missing in metadata. Using all rows."
            )

    if sample_size is not None:
        df_meta = df_meta.head(sample_size)

    records = []

    # Group by file path to minimize file open/close operations
    for file_name, group in df_meta.groupby("file_path"):
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "rb") as f:
            for i, (_, row) in enumerate(group.iterrows()):
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q_text = data.get("question_text", "")
                doc_text = data.get("document_text", "")
                doc_tokens = tokenize(doc_text)

                # We need to find the specific candidate paragraph that contains the short answer
                annotations = data.get("annotations", [])
                valid_entry_found = False

                for ann in annotations:
                    short_answers = ann.get("short_answers", [])
                    if not short_answers:
                        continue

                    # Take the first valid short answer span
                    sa = short_answers[0]
                    s_start = sa["start_token"]
                    s_end = sa["end_token"]

                    # Find the long answer candidate containing this span
                    candidates = data.get("long_answer_candidates", [])
                    for cand in candidates:
                        c_start = cand["start_token"]
                        c_end = cand["end_token"]

                        if c_start <= s_start and c_end >= s_end:
                            # Found the containing paragraph

                            # Extract paragraph text (raw tokens)
                            cand_tokens = doc_tokens[c_start:c_end]

                            # Calculate relative indices within the candidate
                            rel_start = s_start - c_start
                            rel_end = s_end - c_start

                            records.append(
                                {
                                    "example_id": row["example_id"],
                                    "question_text": q_text,
                                    "context_text": " ".join(cand_tokens),
                                    "start_token": rel_start,
                                    "end_token": rel_end,
                                }
                            )
                            valid_entry_found = True
                            break

                    if valid_entry_found:
                        break

    df = pd.DataFrame(records)

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Reader data saved to {output_path}. Shape: {df.shape}")
    return df


def create_reader_dataset(
    metadata_path, output_path, is_train=True, load_cached_data=True, sample_size=None
):
    """
    Orchestrates the creation or loading of the Reader dataset.
    """
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached reader data from {output_path}")
        return pd.read_parquet(output_path)

    return _process_reader_data(metadata_path, output_path, is_train, sample_size)


class ReaderDataset(Dataset):
    """
    PyTorch Dataset for the Short Answer Reader.
    Vectorizes text and maps raw span indices to clean token indices.
    """

    def __init__(self, data_source, vocab, is_train=True):
        """
        Args:
            data_source (pd.DataFrame or str): DataFrame containing data or path to parquet file.
            vocab (dict): Vocabulary mapping token to index.
            is_train (bool): If True, returns targets (start/end indices).
        """
        if isinstance(data_source, str):
            self.data = pd.read_parquet(data_source)
        else:
            self.data = data_source

        self.vocab = vocab
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_text = row["question_text"]
        ctx_text = row["context_text"]

        # 1. Tokenize and Clean Context
        # The context_text comes from the raw document, so it contains HTML tags.
        # The Reader model expects clean text (tags removed).
        raw_ctx_tokens = tokenize(ctx_text)
        clean_ctx_tokens, clean_map = strip_html_tags(raw_ctx_tokens)
        clean_ctx_text = " ".join(clean_ctx_tokens)

        # 2. Vectorize
        q_indices = text_to_indices(q_text, self.vocab, max_len=Config.MAX_Q_LEN)
        ctx_indices = text_to_indices(
            clean_ctx_text, self.vocab, max_len=Config.MAX_CTX_LEN
        )

        q_tensor = torch.tensor(q_indices, dtype=torch.long)
        ctx_tensor = torch.tensor(ctx_indices, dtype=torch.long)

        if self.is_train:
            raw_start = row["start_token"]
            raw_end = row["end_token"]  # Exclusive index

            # 3. Map Raw Targets to Clean Targets
            # We need to find which index in clean_ctx_tokens corresponds to raw_start/raw_end

            # Build explicit map: raw_index -> clean_index
            raw_to_clean = {r_idx: c_idx for c_idx, r_idx in enumerate(clean_map)}

            # Map Start: If raw_start is a tag, advance to next valid token
            curr = raw_start
            while curr < len(raw_ctx_tokens) and curr not in raw_to_clean:
                curr += 1
            target_start = raw_to_clean.get(curr, 0)  # Default to 0 if not found

            # Map End: raw_end is exclusive.
            # If raw_end falls on a tag, we want the clean index of the next valid token
            # (which effectively serves as the exclusive end for the previous valid tokens).
            curr = raw_end
            if curr >= len(raw_ctx_tokens):
                target_end = len(clean_ctx_tokens)
            else:
                while curr < len(raw_ctx_tokens) and curr not in raw_to_clean:
                    curr += 1
                target_end = raw_to_clean.get(curr, len(clean_ctx_tokens))

            # 4. Clamp to Model Input Length
            # Ensure targets fit within the truncated sequence
            target_start = min(target_start, Config.MAX_CTX_LEN - 1)
            target_end = min(target_end, Config.MAX_CTX_LEN - 1)

            # Safety check: start must be <= end
            if target_start > target_end:
                target_end = target_start

            return (
                q_tensor,
                ctx_tensor,
                torch.tensor(target_start, dtype=torch.long),
                torch.tensor(target_end, dtype=torch.long),
            )

        return q_tensor, ctx_tensor


def get_reader_loaders(
    vocab,
    batch_size=Config.READER_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    sample_size=None,
):
    """
    Utility to get DataLoaders for the Reader model training.

    Args:
        vocab (dict): Vocabulary dictionary.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        sample_size (int, optional): Number of samples to use (for debugging).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # 1. Prepare Dataframes
    train_df = create_reader_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.READER_TRAIN_CACHE,
        is_train=True,
        sample_size=sample_size,
    )

    val_df = create_reader_dataset(
        Config.VAL_METADATA_PATH,
        Config.READER_VAL_CACHE,
        is_train=True,
        sample_size=sample_size,
    )

    # 2. Create Datasets
    train_ds = ReaderDataset(train_df, vocab, is_train=True)
    val_ds = ReaderDataset(val_df, vocab, is_train=True)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
